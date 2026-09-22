#!/usr/bin/env python3
"""A-B-A-B prefill bench: agentic-coding traffic driver + server-log analysis.

Drives a single-stream agentic CODING conversation (tool-call loop over real
repo sources, streaming, growing history — the shape of production traffic)
against a running oMLX server, then extracts the matching server-log window
(prefix-cache restores, PLE gather/eviction stats, completions) and fits

    ttft ≈ fixed + suffix_tokens / marginal_rate

so runs across builds (yarn-scaling vs v0.7.0.dev4) compare on marginal
prefill tok/s at matched context sizes, with the n-gram/PLE eviction count
as the residency covariate.

Usage:
  # one run against whatever build is up (label it):
  python scripts/prefill_ab_bench.py run --label yarn-r1 \
      --base-url http://127.0.0.1:1234 --target-tokens 32000

  # after each run (any time): comparison table
  python scripts/prefill_ab_bench.py report

  # standalone extraction over a log window (validation / post-mortem):
  python scripts/prefill_ab_bench.py extract \
      --start "2026-09-18 14:36:40" --end "2026-09-18 14:43:00"

Results append to scripts/prefill_ab.jsonl (one JSON object per run).
Stdlib only. The run nonce in the system prompt guarantees each run's
prefix is cache-unique: no cross-run prefix hits, full intra-run reuse.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = Path.home() / ".omlx" / "logs" / "server.log"
DEFAULT_OUT = Path(__file__).resolve().parent / "prefill_ab.jsonl"
DEFAULT_MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"

# Real repo sources cycled into synthetic tool results (agentic-coding shape).
CORPUS_FILES = [
    "omlx/scheduler.py",
    "omlx/turboquant_kv.py",
    "omlx/patches/turboquant_attention.py",
    "omlx/patches/mlx_vlm_qwen4_exp_compat/yarn_rope.py",
    "omlx/engine/batched.py",
    "omlx/cache/prefix_cache.py",
    "omlx/cache/paged_ssd_cache.py",
    "omlx/settings.py",
    "omlx/server.py",
    "omlx/admin/routes.py",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a slice of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Regex search across repository files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the repository checkout.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_edit",
            "description": "Apply a unified-diff edit to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "diff": {"type": "string"},
                },
                "required": ["path", "diff"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert coding agent working in the oMLX repository (an MLX-based \
LLM server for Apple Silicon). Benchmark run id: {nonce}.

Task: audit the prefill path end to end. Work methodically: read the \
scheduler's chunked-prefill loop, the Qwen4-Exp vendored model, the cache \
subsystem, and propose precise improvements. Always call a tool to \
investigate before answering; cite file paths and line numbers."""


# ---------------------------------------------------------------------------
# Synthetic tool results from real repo sources
# ---------------------------------------------------------------------------


class CorpusCycler:
    """Hands out successive windows of real repo files as tool-result text."""

    def __init__(self):
        self._chunks: list[str] = []
        for rel in CORPUS_FILES:
            path = REPO / rel
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            step = 120
            for start in range(0, len(lines), step):
                window = lines[start : start + step]
                header = f"{rel}:{start + 1}-{start + len(window)}"
                self._chunks.append(
                    header + "\n" + "\n".join(f"{start + 1 + i}: {ln}" for i, ln in enumerate(window))
                )
        self._cursor = 0

    def take_chars(self, n_chars: int) -> str:
        if not self._chunks:
            return "x" * max(n_chars, 1)
        out: list[str] = []
        total = 0
        while total < n_chars:
            chunk = self._chunks[self._cursor % len(self._chunks)]
            self._cursor += 1
            out.append(chunk)
            total += len(chunk)
        text = "\n".join(out)
        return text[: max(n_chars, 512)]


# ---------------------------------------------------------------------------
# Streaming chat client
# ---------------------------------------------------------------------------


def _post_stream(url: str, payload: dict, api_key: str | None, timeout: float):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def chat_turn(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str | None,
    max_tokens: int,
    timeout: float,
) -> dict:
    """One streaming chat completion. Returns timing + reconstructed message."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage = None
    resp = _post_stream(base_url.rstrip("/") + "/v1/chat/completions", payload, api_key, timeout)
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
                calls = delta.get("tool_calls")
                if ttft is None and (piece or calls):
                    ttft = time.perf_counter() - t0
                if piece:
                    content_parts.append(piece)
                for call in calls or []:
                    slot = tool_calls.setdefault(
                        call.get("index", 0),
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    total = time.perf_counter() - t0
    message = {"role": "assistant", "content": "".join(content_parts) or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {
        "message": message,
        "ttft_s": ttft if ttft is not None else total,
        "total_s": total,
        "finish_reason": finish_reason,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Server-log extraction
# ---------------------------------------------------------------------------

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_COMPLETION = re.compile(
    r"Chat completion: model=(\S+), (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), "
    r"prompt: (\d+),.*?stream_model_ttft=([\d.]+)s"
)
_RESTORE = re.compile(
    r"Prefix cache restore for ([0-9a-f-]+): source=(\w+) cached=(\d+) suffix=(\d+) "
    r"blocks=(\d+) lookup=([\d.]+)ms reconstruct=([\d.]+)ms"
)
_PLE = re.compile(r"PLE: warm gather of (\d+) rows took ([\d.]+) ms \(memcpy budget ([\d.]+) ms\)")


def _parse_ts(line: str):
    m = _TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")


def extract_window(log_path: Path, start: datetime, end: datetime) -> dict:
    completions, restores, gathers, evictions, fairness = [], [], [], 0, 0
    if not log_path.exists():
        return {
            "completions": completions, "restores": restores, "gathers": gathers,
            "evictions": evictions, "fairness_reductions": fairness,
        }
    with log_path.open("rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            ts = _parse_ts(line)
            if ts is None or ts < start or ts > end:
                continue
            if (m := _COMPLETION.search(line)):
                completions.append(
                    {"ts": ts.isoformat(), "tokens": int(m.group(2)), "dur_s": float(m.group(3)),
                     "decode_tps": float(m.group(4)), "prompt": int(m.group(5)),
                     "ttft_s": float(m.group(6))}
                )
            elif (m := _RESTORE.search(line)):
                restores.append(
                    {"ts": ts.isoformat(), "rid": m.group(1), "source": m.group(2),
                     "cached": int(m.group(3)), "suffix": int(m.group(4)),
                     "blocks": int(m.group(5)), "reconstruct_ms": float(m.group(7))}
                )
            elif (m := _PLE.search(line)):
                gathers.append({"rows": int(m.group(1)), "ms": float(m.group(2)),
                                "budget_ms": float(m.group(3))})
                if "eviction suspected" in line:
                    evictions += 1
            elif "[fairness] Prefill chunk reduced" in line:
                fairness += 1
    return {
        "completions": completions, "restores": restores, "gathers": gathers,
        "evictions": evictions, "fairness_reductions": fairness,
    }


def pair_restores_with_ttft(server: dict, prompt_tol: int = 256) -> list[dict]:
    """Pair each restore with its completion.

    prompt == cached + suffix exactly, so prefer prompt proximity over raw
    time closeness; mixed-stream windows otherwise glue a small stream's
    restore onto a huge stream's completion that finished nearby.
    """
    pairs = []
    used = set()
    for restore in server["restores"]:
        want = restore["cached"] + restore["suffix"]
        best, best_key = None, None
        for i, comp in enumerate(server["completions"]):
            if i in used:
                continue
            dprompt = abs(comp["prompt"] - want)
            if dprompt > prompt_tol:
                continue
            dt = abs(
                datetime.fromisoformat(comp["ts"]) - datetime.fromisoformat(restore["ts"])
            ).total_seconds()
            if dt > 300:
                continue
            key = (dprompt, dt)
            if best_key is None or key < best_key:
                best, best_key = i, key
        if best is not None:
            used.add(best)
            comp = server["completions"][best]
            pairs.append({"suffix": restore["suffix"], "cached": restore["cached"],
                          "ttft_s": comp["ttft_s"], "prompt": comp["prompt"],
                          "decode_tps": comp["decode_tps"]})
    return pairs


def fit_fixed_marginal(pairs: list[dict], max_cached: int | None = None) -> dict:
    """Least squares ttft = a + b*suffix; max_cached limits the prefix band."""
    pts = [
        (p["suffix"], p["ttft_s"])
        for p in pairs
        if p["suffix"] > 0 and (max_cached is None or p["cached"] <= max_cached)
    ]
    if len(pts) < 3:
        return {"n": len(pts)}
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return {"n": n}
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    out = {"n": n, "fixed_s": round(a, 3)}
    if b > 0:
        out["ms_per_token"] = round(b * 1000, 4)
        out["marginal_tok_s"] = round(1000.0 / (b * 1000), 1)
    return out


def summarize_server(server: dict, fit_max_cached: int | None = None) -> dict:
    pairs = pair_restores_with_ttft(server)
    gathers = server["gathers"]
    over = [g for g in gathers if g["ms"] > g["budget_ms"]]
    return {
        "fit": fit_fixed_marginal(pairs, max_cached=fit_max_cached),
        "pairs": pairs,
        "restores": len(server["restores"]),
        "completions": len(server["completions"]),
        "ple_gathers": len(gathers),
        "ple_over_budget": len(over),
        "ple_evictions": server["evictions"],
        "ple_mean_ms": round(sum(g["ms"] for g in gathers) / len(gathers), 2) if gathers else None,
        "fairness_reductions": server["fairness_reductions"],
        "mean_decode_tps": (
            round(sum(c["decode_tps"] for c in server["completions"]) / len(server["completions"]), 1)
            if server["completions"] else None
        ),
    }


# ---------------------------------------------------------------------------
# run: agentic-coding conversation driver
# ---------------------------------------------------------------------------


def run_bench(args) -> dict:
    api_key = args.api_key or os.environ.get("OMLX_API_KEY") or ""
    nonce = uuid.uuid4().hex[:12]
    corpus = CorpusCycler()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(nonce=nonce)},
        {"role": "user", "content": "Start with omlx/scheduler.py: locate the chunked-prefill "
                                    "loop and summarize the chunk-sizing inputs."},
    ]
    turns = []
    chars_per_token = 3.4
    started = datetime.now()
    print(f"[{args.label}] run {nonce} -> {args.base_url} model={args.model}")
    try:
        while len(turns) < args.max_turns:
            turn = chat_turn(args.base_url, args.model, messages, api_key or None,
                             args.max_tokens, args.timeout)
            prompt_tokens = turn["prompt_tokens"] or 0
            turns.append(turn)
            messages.append(turn["message"])
            decode_tps = None
            if turn["completion_tokens"]:
                gen_time = max(turn["total_s"] - turn["ttft_s"], 1e-6)
                decode_tps = round(turn["completion_tokens"] / gen_time, 1)
            print(
                f"  turn {len(turns):>2}: prompt={prompt_tokens:>7} "
                f"ttft={turn['ttft_s']:6.2f}s total={turn['total_s']:6.2f}s "
                f"gen={turn['completion_tokens']} ({decode_tps} tok/s) "
                f"finish={turn['finish_reason']}"
            )
            if prompt_tokens >= args.target_tokens:
                break

            tool_calls = turn["message"].get("tool_calls") or []
            if tool_calls:
                # Adapt chars/token from the observed prompt growth.
                if len(turns) >= 2:
                    prev_prompt = turns[-2]["prompt_tokens"] or 0
                    added_chars = sum(
                        len(m.get("content") or "") for m in messages
                        if m.get("role") == "tool"
                    )
                    if prev_prompt and added_chars:
                        observed = added_chars / max(prompt_tokens - prev_prompt + 1, 1)
                        if 1.0 < observed < 20.0:
                            chars_per_token = 0.7 * chars_per_token + 0.3 * observed
                per_call = int(args.growth_tokens * chars_per_token / len(tool_calls))
                for call in tool_calls:
                    fn = call["function"]["name"] or "tool"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"] or f"call_{uuid.uuid4().hex[:8]}",
                        "name": fn,
                        "content": corpus.take_chars(per_call),
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": "Continue the audit: inspect the next component and show the "
                               "exact code you would change, with file paths.",
                })
            time.sleep(args.gap)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  !! run aborted: {exc}", file=sys.stderr)
    ended = datetime.now()

    time.sleep(args.settle)
    server = extract_window(Path(args.server_log), started, ended)
    summary = summarize_server(server)
    record = {
        "label": args.label,
        "nonce": nonce,
        "base_url": args.base_url,
        "model": args.model,
        "target_tokens": args.target_tokens,
        "growth_tokens": args.growth_tokens,
        "started": started.isoformat(timespec="seconds"),
        "ended": ended.isoformat(timespec="seconds"),
        "client_turns": [
            {k: t[k] for k in ("ttft_s", "total_s", "prompt_tokens",
                               "completion_tokens", "finish_reason", "started_at")}
            for t in turns
        ],
        "server": {k: v for k, v in summary.items() if k != "pairs"},
        "server_pairs": summary["pairs"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"[{args.label}] fit: {summary['fit']}  evictions={summary['ple_evictions']} "
          f"over-budget={summary['ple_over_budget']}/{summary['ple_gathers']} "
          f"fairness-caps={summary['fairness_reductions']} -> {out}")
    return record


# ---------------------------------------------------------------------------
# report / extract modes
# ---------------------------------------------------------------------------


def report(args):
    records = []
    path = Path(args.out)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        print(f"no runs in {path}")
        return
    header = (f"{'label':<12} {'started':<20} {'turns':>5} {'fixed_s':>8} "
              f"{'marg tok/s':>10} {'evict':>6} {'overb':>6} {'fair':>5} {'dec tps':>8}")
    print(header)
    print("-" * len(header))
    for r in records:
        fit = r["server"]["fit"]
        print(
            f"{r['label']:<12} {r['started']:<20} {len(r['client_turns']):>5} "
            f"{fit.get('fixed_s', '-'):>8} {fit.get('marginal_tok_s', '-'):>10} "
            f"{r['server']['ple_evictions']:>6} {r['server']['ple_over_budget']:>6} "
            f"{r['server']['fairness_reductions']:>5} "
            f"{r['server'].get('mean_decode_tps') or '-':>8}"
        )


def extract_cmd(args):
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
    server = extract_window(Path(args.log), start, end)
    summary = summarize_server(server, fit_max_cached=args.fit_max_cached)
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2))
    for pair in summary["pairs"]:
        print(f"  suffix={pair['suffix']:>6} cached={pair['cached']:>7} "
              f"ttft={pair['ttft_s']:>6.2f}s prompt={pair['prompt']:>7}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="drive one agentic-coding bench conversation")
    run.add_argument("--label", required=True, help="build/run label, e.g. yarn-r1, dev4-r1")
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--api-key", default=None, help="or OMLX_API_KEY env")
    run.add_argument("--target-tokens", type=int, default=32000)
    run.add_argument("--growth-tokens", type=int, default=4000,
                     help="tool-result growth per exchange")
    run.add_argument("--max-turns", type=int, default=24)
    run.add_argument("--max-tokens", type=int, default=1024,
                     help="completion cap per turn (bounds decode time; TTFT unaffected)")
    run.add_argument("--gap", type=float, default=0.5, help="seconds between turns")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--settle", type=float, default=3.0,
                     help="seconds to wait before reading the server log")
    run.add_argument("--server-log", default=str(DEFAULT_LOG))
    run.add_argument("--out", default=str(DEFAULT_OUT))
    run.set_defaults(fn=run_bench)

    rep = sub.add_parser("report", help="comparison table over all runs")
    rep.add_argument("--out", default=str(DEFAULT_OUT))
    rep.set_defaults(fn=report)

    ext = sub.add_parser("extract", help="standalone server-log window analysis")
    ext.add_argument("--log", default=str(DEFAULT_LOG))
    ext.add_argument("--start", required=True, help='"YYYY-MM-DD HH:MM:SS"')
    ext.add_argument("--end", required=True, help='"YYYY-MM-DD HH:MM:SS"')
    ext.add_argument("--fit-max-cached", type=int, default=None,
                     help="fit only pairs with cached <= N (like-with-like in "
                          "mixed-stream windows, e.g. 60000)")
    ext.set_defaults(fn=extract_cmd)

    args = parser.parse_args()
    try:
        result = args.fn(args)
    except BrokenPipeError:  # piped to head & co
        return 0
    return 0 if result is not None or args.cmd != "run" else 1


if __name__ == "__main__":
    sys.exit(main())
