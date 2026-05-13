# SPDX-License-Identifier: Apache-2.0
"""Build supplemental oQ calibration data (code / tool_calling / reasoning).

Downloads from 5 HuggingFace sources, classifies text into code,
tool_calling, and reasoning categories, and samples each to a character
budget using deterministic (hash-sorted) selection.

The multilingual categories (en, ko, zh, ja) are kept from the upstream
oq_calibration_data.json as-is — this script only rebuilds the focus
categories that benefit most from additional calibration diversity.

Extracted results are cached per-dataset in scripts/.calibration_cache/ so
subsequent runs skip the download entirely.

Usage:
    pip install datasets   # one-time
    python scripts/build_calibration_data.py
    python scripts/build_calibration_data.py --no-cache   # force re-download
    python scripts/build_calibration_data.py --dry-run    # print stats only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".calibration_cache"

CODE_MARKERS = re.compile(
    r"(?:^|\s)(?:def |class |function |import |from .+ import |"
    r"#include|public static|console\.\w+|```)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _load_cache(name: str) -> dict[str, list[str]] | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    print(f"  [{name}] loaded from cache ({p.stat().st_size / 1024:.0f} KB)")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(name: str, data: dict[str, list[str]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  [{name}] cached ({p.stat().st_size / 1024:.0f} KB)")


def _cached(name: str, loader, use_cache: bool = True):
    if use_cache:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    result = loader()
    _save_cache(name, result)
    return result


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _has_code(text: str) -> bool:
    return bool(CODE_MARKERS.search(text))


def _flatten_sharegpt(conversations: list[dict], include_thinking: bool = True) -> str:
    parts = []
    for msg in conversations:
        role = msg.get("from") or msg.get("role", "unknown")
        content = msg.get("value") or msg.get("content", "")
        reasoning = msg.get("reasoning_content")

        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role in ("human", "user"):
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role in ("gpt", "assistant"):
            thinking_block = ""
            if include_thinking and reasoning:
                thinking_block = f"<think>\n{reasoning}\n</think>\n"
            parts.append(
                f"<|im_start|>assistant\n{thinking_block}{content}<|im_end|>"
            )
        elif role == "tool":
            parts.append(f"<|im_start|>tool\n{content}<|im_end|>")
    return "\n".join(parts)


def _sample_by_chars(items: list[str], char_budget: int) -> list[str]:
    """Deterministic char-budget sampling: sort by content hash, fill greedily."""
    if sum(len(t) for t in items) <= char_budget:
        return items
    sorted_items = sorted(items, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    picked = []
    total = 0
    for t in sorted_items:
        if total + len(t) > char_budget:
            continue
        picked.append(t)
        total += len(t)
    return picked


# ---------------------------------------------------------------------------
# Per-dataset loaders (return ALL extracted texts, sampling happens later)
# ---------------------------------------------------------------------------


def _download_qwen3_dwq() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading mlx-community/qwen3_dwq_calibration_1332 ...")
    ds = load_dataset(
        "mlx-community/qwen3_dwq_calibration_1332", split="train"
    )
    print(f"  loaded {len(ds)} rows")

    reasoning_texts: list[str] = []

    for row in ds:
        messages = row["messages"]
        flat = _flatten_sharegpt(messages, include_thinking=True)
        if flat and "<think>" in flat:
            reasoning_texts.append(flat)

    print(f"  reasoning: {len(reasoning_texts)} texts extracted")
    return {"reasoning": reasoning_texts}


def _download_mixed_exl() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading Orion-zhen/mixed-exl-calibration ...")
    ds = load_dataset("Orion-zhen/mixed-exl-calibration", split="train")
    print(f"  loaded {len(ds)} rows")

    code_texts: list[str] = []

    for row in ds:
        text = row["content"]
        if text and _has_code(text):
            code_texts.append(text)

    print(f"  code: {len(code_texts)} texts extracted")
    return {"code": code_texts}


def _download_reasoning_exl() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading Orion-zhen/reasoning-exl-calibration (streaming) ...")
    ds = load_dataset(
        "Orion-zhen/reasoning-exl-calibration", split="train", streaming=True
    )

    reasoning_texts: list[str] = []
    code_texts: list[str] = []
    target = 3600

    for row in ds.take(target):
        inp = row.get("input", "")
        out = row.get("output", "")
        combined = f"{inp}\n\n{out}"
        if not combined.strip():
            continue

        if _has_code(combined):
            code_texts.append(combined)
        else:
            reasoning_texts.append(combined)

    result = {"reasoning": reasoning_texts, "code": code_texts}
    for k, v in result.items():
        print(f"  {k}: {len(v)} texts extracted (from {target} scanned)")
    return result


def _download_hermes_function_calling() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading NousResearch/hermes-function-calling-v1 ...")
    subsets = [
        "func_calling_singleturn",
        "func_calling",
        "glaive_func_calling",
        "json_mode_agentic",
        "json_mode_singleturn",
    ]

    all_texts: list[str] = []
    for subset in subsets:
        try:
            ds = load_dataset(
                "NousResearch/hermes-function-calling-v1",
                subset,
                split="train",
            )
            for row in ds:
                conversations = row.get("conversations", [])
                flat = _flatten_sharegpt(conversations, include_thinking=False)
                if flat:
                    all_texts.append(flat)
        except Exception as e:
            print(f"  WARNING: failed to load subset {subset}: {e}")

    result = {"tool_calling": all_texts}
    print(f"  tool_calling: {len(all_texts)} texts extracted")
    return result


def _download_hermes_agent_traces() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading lambda/hermes-agent-reasoning-traces (kimi, streaming) ...")
    ds = load_dataset(
        "lambda/hermes-agent-reasoning-traces",
        "kimi",
        split="train",
        streaming=True,
    )

    tool_texts: list[str] = []
    reasoning_texts: list[str] = []
    pool_size = 1000

    for row in ds.take(pool_size):
        conversations = row.get("conversations", [])
        flat = _flatten_sharegpt(conversations, include_thinking=True)
        if not flat:
            continue
        if "<think>" in flat:
            reasoning_texts.append(flat)
        else:
            tool_texts.append(flat)

    result = {"tool_calling": tool_texts, "reasoning": reasoning_texts}
    for k, v in result.items():
        print(f"  {k} (agent traces): {len(v)} texts extracted")
    print(f"  (from {pool_size} scanned)")
    return result


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _merge(target: dict[str, list[str]], source: dict[str, list[str]]) -> None:
    for k, v in source.items():
        target.setdefault(k, []).extend(v)


def build_calibration_data(use_cache: bool = True,
                           focus_budget: int = 200_000) -> dict[str, list[str]]:
    sources = [
        ("qwen3_dwq", _download_qwen3_dwq),
        ("mixed_exl", _download_mixed_exl),
        ("reasoning_exl", _download_reasoning_exl),
        ("hermes_func", _download_hermes_function_calling),
        ("hermes_agent", _download_hermes_agent_traces),
    ]

    raw: dict[str, list[str]] = {}
    for name, loader in sources:
        _merge(raw, _cached(name, loader, use_cache=use_cache))

    data = {
        "code": _sample_by_chars(raw.get("code", []), focus_budget),
        "tool_calling": _sample_by_chars(raw.get("tool_calling", []), focus_budget),
        "reasoning": _sample_by_chars(raw.get("reasoning", []), focus_budget),
    }

    print("\n--- Final counts ---")
    total_chars = sum(sum(len(t) for t in v) for v in data.values())
    for k in sorted(data.keys()):
        chars = sum(len(t) for t in data[k])
        pct = chars / max(total_chars, 1) * 100
        print(f"  {k}: {len(data[k]):4d} samples, {chars:>8,} chars ({pct:5.1f}%)")
    print(f"  TOTAL: {sum(len(v) for v in data.values())} samples, "
          f"{total_chars:,} chars")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build oQ calibration data from curated sources"
    )
    parser.add_argument(
        "--output",
        default="omlx/oq_calibration_data.json",
        help="Output JSON path (default: omlx/oq_calibration_data.json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download, ignore cached extracts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing the file",
    )
    args = parser.parse_args()

    focus_data = build_calibration_data(use_cache=not args.no_cache)

    out = Path(args.output)

    # Merge with existing JSON to preserve multilingual categories
    existing: dict[str, list[str]] = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"\nMerging with existing {out} ({len(existing)} categories)")

    MULTILINGUAL_KEYS = ("en", "ko", "zh", "ja")
    merged = {}
    for k in MULTILINGUAL_KEYS:
        if k in existing:
            merged[k] = existing[k]
    merged.update(focus_data)

    # Canonical key order
    ordered = {k: merged[k] for k in
               ("code", "en", "ko", "zh", "ja", "tool_calling", "reasoning")
               if k in merged}

    total_chars = sum(sum(len(t) for t in v) for v in ordered.values())
    print(f"\n--- Merged output ---")
    for k in ordered:
        chars = sum(len(t) for t in ordered[k])
        pct = chars / max(total_chars, 1) * 100
        src = "existing" if k in MULTILINGUAL_KEYS else "rebuilt"
        print(f"  {k}: {len(ordered[k]):4d} samples, {chars:>8,} chars "
              f"({pct:5.1f}%)  [{src}]")
    print(f"  TOTAL: {sum(len(v) for v in ordered.values())} samples, "
          f"{total_chars:,} chars")

    if args.dry_run:
        print("Dry run — not writing file.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Written to {out} ({out.stat().st_size / 1024:.0f} KB on disk)")


if __name__ == "__main__":
    main()
