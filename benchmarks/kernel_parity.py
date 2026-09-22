#!/usr/bin/env python3
"""L1: cross-stack greedy-parity capture/compare for kernel compatibility.

Same build code, two kernel stacks (app bundle with compiled NAX/metallib
kernels vs dev checkout with runtime-compiled Metal fallbacks), identical
fixed prompts, temperature 0. Deterministic greedy decode => any logit
divergence beyond bf16 reduction-order drift eventually flips a token.
Metrics: exact match, first divergence (chars), agreement ratio over the
generated span. Multiple prompts so an early flip on one near-tie doesn't
dominate the verdict.

  capture: python benchmarks/kernel_parity.py capture --label app --base-url URL
  compare: python benchmarks/kernel_parity.py compare --a app --b dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "parity"
MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
MAX_TOKENS = 128

# Deterministic prompt fixtures built from repo sources (stable hashes).
_PROMPT_SOURCES = [
    ("omlx/scheduler.py", 3680, 3760),
    ("omlx/turboquant_kv.py", 240, 330),
    ("omlx/patches/turboquant_attention.py", 320, 400),
    ("omlx/cache/paged_ssd_cache.py", 250, 330),
]


def _fixture(kind: str) -> str:
    if kind == "code2k":
        path, a, b = _PROMPT_SOURCES[0]
        body = "\n".join((REPO / path).read_text(errors="replace").splitlines()[a:b])
        return (
            "Below is a code slice. Explain step by step what it does and why "
            "each guard exists.\n\n" + body
        )
    if kind == "code8k":
        parts = []
        for path, a, b in _PROMPT_SOURCES:
            lines = (REPO / path).read_text(errors="replace").splitlines()[a:b]
            parts.append(f"### {path}\n" + "\n".join(lines))
        filler = "\n".join(parts) * 4
        return (
            "Audit the following excerpts for correctness issues; list "
            "file:line and a one-line rationale for each finding.\n\n"
            + filler[: 27000]
        )
    if kind == "mixed":
        path, a, b = _PROMPT_SOURCES[1]
        body = "\n".join((REPO / path).read_text(errors="replace").splitlines()[a:b])
        return (
            "Summarize the invariants this module maintains, then propose one "
            "additional regression test that would fail if the batch-offset "
            "tracking broke.\n\n" + body
        )
    raise ValueError(kind)


FIXTURES = ["code2k", "code8k", "mixed"]


def _post(url: str, payload: dict, api_key: str | None, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def capture(args) -> int:
    api_key = args.api_key or os.environ.get("OMLX_API_KEY") or None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for kind in FIXTURES:
        prompt = _fixture(kind)
        if args.nonce:
            prompt = f"[parity run {args.nonce}]\n" + prompt
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": False,
        }
        t0 = time.perf_counter()
        resp = _post(args.base_url.rstrip("/") + "/v1/chat/completions",
                     payload, api_key, args.timeout)
        dt = time.perf_counter() - t0
        choice = resp["choices"][0]
        message = choice["message"]
        text = (message.get("reasoning_content") or "") + "\x00" + (message.get("content") or "")
        record = {
            "fixture": kind,
            "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage"),
            "wall_s": round(dt, 2),
            "label": args.label,
            "base_url": args.base_url,
            "model": args.model,
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(record)
        print(f"  {kind:<7} prompt_sha={record['prompt_sha']} wall={dt:6.2f}s "
              f"finish={record['finish_reason']} chars={len(text)}")
    out = OUT_DIR / f"{args.label}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    return 0


def _first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) == len(b) else n


def compare(args) -> int:
    a = json.loads((OUT_DIR / f"{args.a}.json").read_text())
    b = json.loads((OUT_DIR / f"{args.b}.json").read_text())
    print(f"{'fixture':<8} {'sha match':<10} {'exact':<6} {'diverg@':>8} "
          f"{'agree%':>7}  verdict")
    worst = 1.0
    for ra, rb in zip(a, b):
        assert ra["fixture"] == rb["fixture"]
        sha_ok = ra["prompt_sha"] == rb["prompt_sha"]
        ta, tb = ra["text"], rb["text"]
        div = _first_divergence(ta, tb)
        agree = div / max(len(ta), len(tb), 1)
        exact = ta == tb
        if not exact:
            worst = min(worst, agree)
        verdict = "MATCH" if exact else ("ok-drift" if agree >= args.min_agree else "DIVERGED")
        print(f"{ra['fixture']:<8} {str(sha_ok):<10} {str(exact):<6} {div:>8} "
              f"{100 * agree:>6.1f}%  {verdict}")
    print(f"\nworst agreement: {100 * worst:.1f}%  (threshold {100 * args.min_agree:.0f}%)")
    return 0 if worst >= args.min_agree else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--label", required=True)
    cap.add_argument("--nonce", default=None,
                     help="unique tag prepended to every fixture prompt; both "
                          "arms of a comparison must share it, and it must be "
                          "fresh vs any cached prefix")
    cap.add_argument("--base-url", default="http://127.0.0.1:1234")
    cap.add_argument("--model", default=MODEL)
    cap.add_argument("--api-key", default=None)
    cap.add_argument("--timeout", type=float, default=600.0)
    cap.set_defaults(fn=capture)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--a", required=True)
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--min-agree", type=float, default=0.75,
                      help="worst-case char agreement before verdict DIVERGED")
    cmp_.set_defaults(fn=compare)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
