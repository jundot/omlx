#!/usr/bin/env python3
"""L2: context-factor sweep — behavioral continuity across YaRN factors.

For each Max Context Window setting (native / 2x / ~3.85x), updates the
per-model setting via the admin API (which auto-reloads the engine when the
rope horizon moves), warms the engine, then runs:
  1. a small deterministic QA set (temp 0) — garbage/NaN detection and
     quality continuity across factors;
  2. kernel_parity captures labeled f-<tag> — so first-token behavior can
     be diffed pairwise afterwards.

Legitimate rope-scaling effects move outputs gently; a kernel range
assumption shows up as a discontinuity or garbage specifically when
mscale != 1.0 (the 2x/3.85x rows).

  python benchmarks/mscale_sweep.py --base-url http://127.0.0.1:1234
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
SWEEP = [(262144, "native"), (524288, "2x"), (1010000, "3.85x")]
QA = [
    ("What is the capital of France? Answer with one word.", "Paris"),
    ("Compute 17 * 23. Answer with only the number.", "391"),
    ("What language is primarily spoken in Brazil? One word.", "ortuguese"),
    ("In Python, what does `def f(x): return x * 2` return for f(21)? "
     "Answer with only the number.", "42"),
    ("Name the three primary colors of light. Answer as a comma list.", "reen"),
]


def _post(url: str, payload: dict, timeout: float, method: str = "POST") -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body) if body else {}


def put_context(base: str, model: str, ctx: int, timeout: float = 60) -> None:
    url = f"{base.rstrip('/')}/admin/api/models/{model}/settings"
    try:
        _post(url, {"max_context_window": ctx}, timeout, method="PUT")
        print(f"  settings: max_context_window={ctx}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise SystemExit(f"settings PUT failed ({exc.code}): {detail}") from exc


def warm(base: str, model: str, timeout: float = 600) -> float:
    t0 = time.perf_counter()
    _post(base.rstrip("/") + "/v1/chat/completions",
          {"model": model, "messages": [{"role": "user", "content": "Reply OK."}],
           "max_tokens": 8, "temperature": 0, "stream": False}, timeout)
    return time.perf_counter() - t0


def qa_one(base: str, model: str, question: str, timeout: float = 300) -> str:
    resp = _post(base.rstrip("/") + "/v1/chat/completions",
                 {"model": model, "messages": [{"role": "user", "content": question}],
                  "max_tokens": 512, "temperature": 0, "stream": False}, timeout)
    msg = resp["choices"][0]["message"]
    return ((msg.get("reasoning_content") or "") + " " + (msg.get("content") or "")).strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:1234")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--only", default=None, help="single tag to run, e.g. 2x")
    p.add_argument("--out", default=str(HERE / "mscale_sweep.jsonl"))
    args = p.parse_args()

    sweep = [s for s in SWEEP if args.only in (None, s[1])]
    results = []
    for ctx, tag in sweep:
        print(f"== factor {tag} (max_context_window={ctx}) ==")
        put_context(args.base_url, args.model, ctx)
        reload_s = warm(args.base_url, args.model)
        print(f"  warm/reload: {reload_s:.1f}s")
        qa_results = []
        for question, expect in QA:
            answer = qa_one(args.base_url, args.model, question)
            ok = expect in answer
            qa_results.append({"q": question[:40], "ok": ok,
                               "tail": answer[-80:].replace("\n", " ")})
            print(f"  QA {'PASS' if ok else 'FAIL'}: {question[:48]!r} -> ...{answer[-60:]!r}")
        cap = subprocess.run(
            [sys.executable, str(HERE / "kernel_parity.py"), "capture",
             "--label", f"f-{tag}", "--base-url", args.base_url, "--model", args.model],
            capture_output=True, text=True,
        )
        print(cap.stdout.strip() or cap.stderr.strip())
        results.append({"tag": tag, "ctx": ctx, "reload_s": round(reload_s, 1),
                        "qa_pass": sum(r["ok"] for r in qa_results),
                        "qa_total": len(qa_results), "qa": qa_results})

    with Path(args.out).open("a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    passed = all(r["qa_pass"] == r["qa_total"] for r in results)
    print(f"\nsweep complete: {'all QA passed' if passed else 'QA FAILURES present'} "
          f"-> {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
