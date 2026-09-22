#!/usr/bin/env python3
"""L3: long-context needle retrieval on the kernel-equipped build.

Builds a prompt of ~--size-tokens with THREE passphrase needles by
default — one inside the first ~100k real tokens, one mid-context, one
inside the last ~100k — asks for all three back in order, and reports
prefill timing, per-needle correctness, and the server-log covariates
over the run window (PLE evictions, memory-guard rejections, indexer
capacity). Per-needle verdicts map recall loss to context regions
(e.g. the packed-write boundary relative to the pressure-fold point).
Pass --depth to fall back to a single mid needle. Validates: no int32/
grid overflow past the native 262k horizon, QSA block selection still
finds single sentences at depth, admission and memory guards behave at
extreme context.

  python scripts/needle_run.py --label app-1m-3n --size-tokens 1380000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = Path.home() / ".omlx" / "logs" / "server.log"
MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
CHARS_PER_TOKEN = 3.4
# --size-tokens over-provisions ~1.37x vs real prompt tokens (measured:
# 1,050,000 -> 766,934), so 100k real tokens ~ 137k size units.
SIZE_TO_REAL_TOKENS = 0.73
FILLER_FILES = [
    "omlx/scheduler.py",
    "omlx/cache/prefix_cache.py",
    "omlx/server.py",
    "omlx/engine/batched.py",
    "omlx/settings.py",
    "omlx/cache/paged_ssd_cache.py",
]


def _filler_text(target_chars: int, nonce: str) -> str:
    chunks = []
    for rel in FILLER_FILES:
        text = (REPO / rel).read_text(errors="replace")
        chunks.append(f"== {rel} (excerpt run {nonce}) ==\n" + text)
    body = "\n\n".join(chunks)
    repeats = target_chars // len(body) + 1
    return (body * repeats)[:target_chars]


def _post_stream(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    return urllib.request.urlopen(req, timeout=timeout)


def run_needle(args) -> int:
    if args.depth is not None:
        depths = [args.depth]
    else:
        edge = min(
            args.edge_tokens / (args.size_tokens * SIZE_TO_REAL_TOKENS), 0.25
        )
        depths = [edge, 0.5, 1.0 - edge]
    needles, passphrases = [], []
    base_nonce = uuid.uuid4().hex[:8].upper()
    for i, depth in enumerate(depths):
        nonce = base_nonce if i == 0 else uuid.uuid4().hex[:8].upper()
        passphrase = f"ZQVK-{nonce}"
        passphrases.append(passphrase)
        needles.append(
            (
                depth,
                f"\n\n[SYSTEM NOTE {nonce}] IMPORTANT: The magic passphrase "
                f"is {passphrase}. Repeat it verbatim when asked, exactly "
                "as shown.\n\n",
            )
        )
    total_chars = int(args.size_tokens * CHARS_PER_TOKEN)
    filler = _filler_text(total_chars, base_nonce)
    # Splice deepest-first so earlier insertions keep their offsets.
    prompt = filler
    for depth, needle in sorted(needles, key=lambda d: -d[0]):
        at = int(len(filler) * depth)
        prompt = prompt[:at] + needle + prompt[at:]
    if len(passphrases) == 1:
        question = (
            "Question: What is the exact magic passphrase stated in the "
            "system note earlier above? Reply with only the passphrase, "
            "nothing else."
        )
    else:
        question = (
            f"Question: The system notes earlier above state {len(passphrases)} "
            "distinct magic passphrases. Reply with all of them verbatim, in "
            "the order they appeared, one per line, and nothing else."
        )
    messages = [{"role": "user", "content": prompt + "\n\n" + question}]
    payload = {
        "model": args.model,
        "messages": messages,
        # The API requires max_tokens; 32k is a no-op ceiling (thinking is
        # bounded by the model's own thinking_budget_tokens).
        "max_tokens": args.max_tokens or 32_768,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = datetime.now()
    print(
        f"[{args.label}] nonces={[p.split('-', 1)[1] for p in passphrases]} "
        f"depths={[round(d, 3) for d in depths]} "
        f"target≈{args.size_tokens} tok"
    )
    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    usage = None
    resp = _post_stream(args.base_url.rstrip("/") + "/v1/chat/completions", payload,
                        args.timeout)
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            if evt.get("usage"):
                usage = evt["usage"]
            for choice in evt.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(piece)
    wall = time.perf_counter() - t0
    answer = "".join(text_parts)
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    found = [passphrase in answer for passphrase in passphrases]
    ok = all(found)
    ended = datetime.now()
    time.sleep(args.settle)

    covariates = {"evictions": 0, "guard_rejections": 0, "indexer_reserved": None,
                  "ttft_server": None}
    log = Path(args.server_log)
    if log.exists():
        with log.open("rb") as fh:
            for raw in fh:
                line = raw.decode("utf-8", "replace")
                ts = line[:19]
                if ts < started.strftime("%Y-%m-%d %H:%M:%S") or ts > ended.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ):
                    continue
                if "eviction suspected" in line:
                    covariates["evictions"] += 1
                if "memory limit exceeded" in line or "prefill rejected" in line:
                    covariates["guard_rejections"] += 1
                if (m := re.search(r"Reserved QSA indexer capacity for (\d+)", line)):
                    covariates["indexer_reserved"] = int(m.group(1))
                if (m := re.search(r"prompt: (\d+), .*stream_model_ttft=([\d.]+)s", line)):
                    covariates["ttft_server"] = float(m.group(2))

    rate = prompt_tokens / (covariates["ttft_server"] or ttft or wall) if prompt_tokens else 0
    record = {
        "label": args.label,
        "nonces": [p.split("-", 1)[1] for p in passphrases],
        "depths": [round(d, 4) for d in depths],
        "target_tokens": args.size_tokens, "prompt_tokens": prompt_tokens,
        "ttft_client_s": round(ttft or wall, 2),
        "ttft_server_s": covariates["ttft_server"],
        "prefill_tok_s": round(rate, 1),
        "wall_s": round(wall, 2),
        "needles_found": found,
        "passphrase_found": ok,
        "answer": answer.strip()[:240],
        "covariates": covariates,
        "started": started.isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    verdict = " ".join(
        f"d{d:.2f}:{'OK' if f else 'MISS'}" for d, f in zip(depths, found)
    )
    print(f"[{args.label}] {'PASS' if ok else 'FAIL'} — {verdict}; "
          f"prefill ≈ {rate:.0f} tok/s")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:1234")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--size-tokens", type=int, default=300_000)
    p.add_argument("--depth", type=float, default=None,
                   help="single-needle override; default plants 3 needles")
    p.add_argument("--edge-tokens", type=float, default=100_000,
                   help="real-token inset for the first/last needles")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="omit by default: thinking models need an uncapped budget",
    )
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--server-log", default=str(DEFAULT_LOG))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "needle_results.jsonl"))
    args = p.parse_args()
    return run_needle(args)


if __name__ == "__main__":
    sys.exit(main())
