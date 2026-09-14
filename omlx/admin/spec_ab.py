# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark for n-gram lookup drafting (plan v3, F1.3).

Runs N identical chat completions against this server's own OpenAI endpoint
with the n-gram drafter ON, then N with it OFF, and reports mean and stddev
of tokens/second per arm. The drafter toggle is a process-global read on
every draft cycle, so arms flip live — no engine reload between them.

Single-flight and synchronous inside a worker thread; results poll by id.
The prompt is the self-similar code-rewrite workload the drafter targets,
sampled per the model card (the operator's standing rule: never greedy
by default in benchmarks).
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.request
import uuid
from typing import Any, Optional

_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()

_CODE = '''def process_orders(orders):
    total = 0
    for order in orders:
        if order.status == "paid":
            total += order.amount
        elif order.status == "refunded":
            total -= order.amount
        elif order.status == "pending":
            continue
    return total

def summarize(orders):
    total = process_orders(orders)
    return {"count": len(orders), "total": total}'''

_PROMPT = (
    "Here is a Python module:\n\n```python\n" + _CODE + "\n```\n\n"
    "Rewrite the whole module EXACTLY as-is, changing only the variable "
    "name `total` to `balance` everywhere. Output only the code block."
)


_NOVEL_TOPICS = [
    "a lighthouse keeper who collects tide sounds",
    "the last tram ride across a city being renamed",
    "a cartographer mapping a river that moves at night",
    "two beekeepers arguing about the color of winter",
    "an archivist who files smells instead of letters",
    "a bridge painter who never saw the far bank",
    "the night shift at a museum of unfinished machines",
    "a typesetter setting the alphabet of a dying language",
    "a ferry pilot crossing between two time zones daily",
    "an astronomer cataloguing clouds instead of stars",
    "a locksmith retiring on an island without doors",
    "the gardener of a rooftop no one can reach",
]


def _novel_prompt(seq: int) -> str:
    # a fresh topic per request keeps every arm on unseen content
    topic = _NOVEL_TOPICS[seq % len(_NOVEL_TOPICS)]
    return (
        f"Write a short original story (about 350 words) about {topic}. "
        f"Do not repeat sentences. Variation seed: {uuid.uuid4().hex[:8]}."
    )


def _one_request(port: int, api_key: str, model_id: str, max_tokens: int,
                 prompt: str = _PROMPT) -> dict:
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.loads(resp.read())
    wall = time.perf_counter() - t0
    usage = payload.get("usage") or {}
    tokens = int(usage.get("completion_tokens") or 0)
    gen_s = usage.get("generation_duration") or usage.get("total_time") or wall
    return {
        "tokens": tokens,
        "seconds": round(float(gen_s), 3),
        "tok_s": round(tokens / float(gen_s), 2) if gen_s else 0.0,
    }


def _arm_summary(samples: list[dict]) -> dict:
    rates = [s["tok_s"] for s in samples]
    return {
        "runs": samples,
        "mean_tok_s": round(statistics.fmean(rates), 2) if rates else None,
        "stdev_tok_s": (
            round(statistics.stdev(rates), 2) if len(rates) > 1 else None
        ),
    }


def _worker(run: dict, port: int, api_key: str) -> None:
    from ..patches.mlx_lm_mtp import (
        get_ngram_spec_params,
        is_ngram_spec_enabled,
        set_ngram_spec,
    )

    flip = run.get("flip", "enabled")
    original = is_ngram_spec_enabled()
    original_freq = get_ngram_spec_params()[3]
    if flip == "freq_rule":
        arms = (("freq_on", True), ("freq_off", False))
    else:
        arms = (("ngram_on", True), ("ngram_off", False))
    novel = run.get("workload") == "novel"
    seq = 0
    try:
        # warmup: one request outside the tally so both arms start hot
        _one_request(port, api_key, run["model_id"], run["max_tokens"])
        for arm, enabled in arms:
            if flip == "freq_rule":
                # drafter stays ON; only the copy-length rule flips
                set_ngram_spec(True, freq_rule=enabled)
            else:
                set_ngram_spec(enabled)
            samples = []
            for i in range(run["repeats"]):
                prompt = _novel_prompt(seq) if novel else _PROMPT
                seq += 1
                samples.append(
                    _one_request(port, api_key, run["model_id"],
                                 run["max_tokens"], prompt)
                )
                run["progress"] = f"{arm} {i + 1}/{run['repeats']}"
            run["results"][arm] = _arm_summary(samples)
        on = run["results"][arms[0][0]]["mean_tok_s"]
        off = run["results"][arms[1][0]]["mean_tok_s"]
        if on and off:
            run["results"]["gain_pct"] = round((on / off - 1) * 100, 1)
        run["status"] = "done"
    except Exception as exc:  # noqa: BLE001 — surfaced to the poller
        run["status"] = "error"
        run["error"] = str(exc)
    finally:
        set_ngram_spec(original, freq_rule=original_freq)


def start(model_id: str, port: int, api_key: str, repeats: int = 5,
          max_tokens: int = 400, flip: str = "enabled",
          workload: str = "rewrite") -> dict:
    with _LOCK:
        active = next(
            (r for r in _RUNS.values() if r["status"] == "running"), None
        )
        if active is not None:
            return {"error": "spec-ab run already active", "id": active["id"]}
        run = {
            "id": uuid.uuid4().hex[:12],
            "model_id": model_id,
            "repeats": max(2, min(int(repeats), 20)),
            "max_tokens": max(64, min(int(max_tokens), 2048)),
            "flip": flip if flip in ("enabled", "freq_rule") else "enabled",
            "workload": workload if workload in ("rewrite", "novel") else "rewrite",
            "status": "running",
            "progress": "warmup",
            "results": {},
            "started_ts": time.time(),
        }
        _RUNS[run["id"]] = run
    threading.Thread(
        target=_worker, args=(run, port, api_key), daemon=True
    ).start()
    return {"id": run["id"], "status": "running"}


def get(run_id: str) -> Optional[dict]:
    return _RUNS.get(run_id)
