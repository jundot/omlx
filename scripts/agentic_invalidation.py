#!/usr/bin/env python3
"""Agentic prefix-refill test: warm cache + harness-style breakage.

Phase A (build): grow a multi-turn agentic conversation to roughly
--target-tokens with the prefix cache ON, planting a unique passphrase
marker in the first turn, a middle turn, and the penultimate turn.
Each turn's incremental prefill should hit the warm prefix; per-turn
TTFT is recorded as evidence.

Phase B (recall): NO server-side invalidation. The usual source of
unwanted prefix refills is the harness editing the request — changing
the system prompt, summarizing an early turn, truncating thinking —
so the breakage is applied client-side to the built conversation:

  --break-mode system (default): swap the system prompt. Divergence at
    token 0: nothing matches the warm prefix and the FULL context cold
    re-prefills while the old cache stays resident — the maximal-
    pressure refill, with every marker intact and recall expected.
  --break-mode early: summarize the first user turn. Refill from the
    first turn on; the turn-0 marker is genuinely gone, so only the
    later markers are expected back.
  --break-mode none: control run; should hit the warm prefix (large
    cached_tokens, tiny TTFT).

  python scripts/agentic_invalidation.py build  --label ag-600k --target-tokens 600000
  python scripts/agentic_invalidation.py recall --label ag-600k --break-mode system
"""

from __future__ import annotations

import argparse
import json
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
# --target-tokens over-provisions ~1.37x vs real prompt tokens (measured),
# so 100k real tokens ~ 137k size units (same convention as needle_run).
SIZE_TO_REAL_TOKENS = 0.73
FILLER_FILES = [
    "omlx/scheduler.py",
    "omlx/cache/prefix_cache.py",
    "omlx/server.py",
    "omlx/engine/batched.py",
    "omlx/settings.py",
    "omlx/cache/paged_ssd_cache.py",
]
CONV_DIR = Path(__file__).resolve().parent / "agentic_conversations"


def _corpus() -> str:
    chunks = []
    for rel in FILLER_FILES:
        text = (REPO / rel).read_text(errors="replace")
        chunks.append(f"== {rel} ==\n" + text)
    return "\n\n".join(chunks)


def _post_stream(url: str, payload: dict, timeout: float):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _stream_completion(base_url: str, payload: dict, timeout: float):
    """Run one streaming completion; return (answer, ttft, wall, usage)."""
    started = time.perf_counter()
    ttft = None
    parts = []
    usage = None
    resp = _post_stream(
        base_url.rstrip("/") + "/v1/chat/completions", payload, timeout
    )
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
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    parts.append(piece)
    return "".join(parts), ttft, time.perf_counter() - started, usage


def build(args) -> int:
    corpus = _corpus()
    total_chars = int(args.target_tokens * CHARS_PER_TOKEN)
    growth_chars = int(args.growth_tokens * CHARS_PER_TOKEN)
    n_turns = max(6, total_chars // growth_chars)
    # Needle-convention depths: first/last markers inside ~edge-tokens of
    # the real-token edges, one mid-context (Victor's 3-needle probe).
    edge_frac = min(
        args.edge_tokens / max(1.0, args.target_tokens * SIZE_TO_REAL_TOKENS),
        0.25,
    )
    marker_turns = {
        max(0, min(n_turns - 1, int(f * n_turns)))
        for f in (edge_frac, 0.5, 1.0 - edge_frac)
    }
    passphrases = {}
    messages = [
        {
            "role": "system",
            "content": "You are a terse assistant. Answer with one word.",
        }
    ]
    turn_stats = []
    pos = 0
    for turn in range(n_turns):
        chunk = corpus[pos : pos + growth_chars]
        pos = (pos + growth_chars) % max(1, len(corpus) - growth_chars)
        content = f"[Turn {turn} context dump]\n{chunk}\n\n"
        if turn in marker_turns:
            nonce = uuid.uuid4().hex[:8].upper()
            passphrases[turn] = f"ZQVK-{nonce}"
            content += (
                f"[SYSTEM NOTE {nonce}] IMPORTANT: The magic passphrase for "
                f"turn {turn} is {passphrases[turn]}. Repeat it verbatim "
                "when asked, exactly as shown.\n\n"
            )
        content += "Question: reply with the single word DONE."
        messages.append({"role": "user", "content": content})
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        answer, ttft, wall, usage = _stream_completion(
            args.base_url, payload, args.timeout
        )
        prompt_tokens = (usage or {}).get("prompt_tokens") or 0
        turn_stats.append(
            {
                "turn": turn,
                "prompt_tokens": prompt_tokens,
                "ttft_s": round(ttft or wall, 2),
                "wall_s": round(wall, 2),
            }
        )
        print(
            f"[build {args.label}] turn {turn}/{n_turns - 1}: "
            f"prompt={prompt_tokens} ttft={ttft or wall:.1f}s",
            flush=True,
        )
        messages.append({"role": "assistant", "content": answer.strip() or "DONE"})
        time.sleep(args.gap)

    CONV_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "label": args.label,
        "model": args.model,
        "n_turns": n_turns,
        "passphrases": {str(k): v for k, v in passphrases.items()},
        "marker_turns": sorted(marker_turns),
        "marker_depths": [round(t / n_turns, 4) for t in sorted(marker_turns)],
        "messages": messages,
        "turn_stats": turn_stats,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = CONV_DIR / f"{args.label}.json"
    out.write_text(json.dumps(record))
    print(f"[build {args.label}] saved {out} ({n_turns} turns, "
          f"markers {sorted(marker_turns)}: {list(passphrases.values())})")
    return 0


def _break_conversation(messages: list[dict], mode: str) -> list[dict]:
    """Harness-style mutations that invalidate the server-side prefix."""
    messages = [dict(m) for m in messages]
    if mode == "system":
        messages[0] = {
            "role": "system",
            "content": "You are a precise assistant. Answer exactly.",
        }
    elif mode == "early":
        for message in messages:
            if message["role"] == "user":
                tail = message["content"]
                at = tail.find("Question:")
                message["content"] = (
                    "[Earlier context summarized by the harness: the user "
                    "shared repository excerpts and standing instructions.]"
                    "\n\n" + (tail[at:] if at >= 0 else "")
                )
                break
    return messages


def recall(args) -> int:
    conv = json.loads((CONV_DIR / f"{args.label}.json").read_text())
    passphrases = conv["passphrases"]
    messages = _break_conversation(conv["messages"], args.break_mode)
    order = sorted(int(t) for t in passphrases)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Question: The system notes above state {len(passphrases)} "
                "distinct magic passphrases, one per noted turn. Reply with "
                "all of them verbatim, in turn order (turn "
                + ", ".join(str(t) for t in order)
                + "), one per line, and nothing else."
            ),
        }
    )
    payload = {
        "model": conv["model"],
        "messages": messages,
        # The API requires max_tokens; 32k is a no-op ceiling (thinking is
        # bounded by the model's own thinking_budget_tokens, 8192).
        "max_tokens": args.max_tokens or 32_768,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = datetime.now()
    print(
        f"[recall {args.label}] break-mode={args.break_mode} "
        f"starting {started:%H:%M:%S}"
    )
    answer, ttft, wall, usage = _stream_completion(
        args.base_url, payload, args.timeout
    )
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    cached_tokens = ((usage or {}).get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    found = {t: (p in answer) for t, p in passphrases.items()}
    if args.break_mode == "early":
        expected = {t: int(t) != order[0] for t in passphrases}
    else:
        expected = {t: True for t in passphrases}
    ok = all(found[t] for t in passphrases if expected[t])
    ended = datetime.now()
    time.sleep(args.settle)

    covariates = {"evictions": 0, "guard_rejections": 0}
    log = Path(args.server_log)
    if log.exists():
        with log.open("rb") as fh:
            for raw in fh:
                line = raw.decode("utf-8", "replace")
                ts = line[:19]
                if (
                    ts < started.strftime("%Y-%m-%d %H:%M:%S")
                    or ts > ended.strftime("%Y-%m-%d %H:%M:%S")
                ):
                    continue
                if "eviction suspected" in line:
                    covariates["evictions"] += 1
                if "memory limit exceeded" in line or "prefill rejected" in line:
                    covariates["guard_rejections"] += 1

    record = {
        "label": args.label,
        "phase": "recall",
        "break_mode": args.break_mode,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "ttft_s": round(ttft or wall, 2),
        "wall_s": round(wall, 2),
        "prefill_tok_s": (
            round((prompt_tokens - (cached_tokens or 0)) / (ttft or wall), 1)
            if prompt_tokens
            else 0
        ),
        "passphrases_found": found,
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
        f"t{t}:{'OK' if hit else 'MISS'}" for t, hit in found.items()
    )
    print(
        f"[{args.label}] {'PASS' if ok else 'FAIL'} — refill recall {verdict}; "
        f"cached={cached_tokens} prefill ≈ {record['prefill_tok_s']:.0f} tok/s"
    )
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:1234")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--server-log", default=str(DEFAULT_LOG))
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "agentic_results.jsonl"),
    )
    sub = p.add_subparsers(dest="phase", required=True)
    b = sub.add_parser("build", help="grow the multi-turn conversation")
    b.add_argument("--target-tokens", type=int, default=600_000)
    b.add_argument("--growth-tokens", type=int, default=20_000)
    b.add_argument(
        "--edge-tokens",
        type=float,
        default=100_000,
        help="real-token inset for the first/last markers",
    )
    b.add_argument("--gap", type=float, default=0.5)
    b.set_defaults(func=build)
    r = sub.add_parser("recall", help="broken-prefix refill and marker recall")
    r.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="omit by default: thinking models need an uncapped budget "
        "(a 256 cap truncated answers mid-passphrase and faked recall "
        "failures)",
    )
    r.add_argument(
        "--break-mode",
        choices=["system", "early", "none"],
        default="system",
        help="client-side prefix invalidation style (default: system swap)",
    )
    r.set_defaults(func=recall)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
