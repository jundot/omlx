# SPDX-License-Identifier: Apache-2.0
"""Build oQ calibration data from curated HuggingFace datasets + bartowski v3.

Downloads from 7 sources, extracts/formats text per category, samples to
target counts, and writes oq_calibration_data.json.

Extracted results are cached per-dataset in scripts/.calibration_cache/ so
subsequent runs skip the download entirely.

Usage:
    pip install datasets   # one-time
    python scripts/build_calibration_data.py
    python scripts/build_calibration_data.py --seed 42
    python scripts/build_calibration_data.py --no-cache   # force re-download
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".calibration_cache"

BARTOWSKI_GIST_URL = (
    "https://gist.githubusercontent.com/bartowski1182/"
    "eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
)

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


def _is_english(text: str) -> bool:
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1) < 0.05


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


def _sample(items: list[str], n: int) -> list[str]:
    if len(items) <= n:
        return items
    return random.sample(items, n)


# ---------------------------------------------------------------------------
# Per-dataset loaders (return ALL extracted texts, sampling happens later)
# ---------------------------------------------------------------------------


def _download_bartowski() -> dict[str, list[str]]:
    print("Downloading bartowski calibration_datav3.txt ...")
    with urllib.request.urlopen(BARTOWSKI_GIST_URL) as resp:
        raw = resp.read().decode("utf-8")
    print(f"  downloaded {len(raw)} chars")
    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
    print(f"  bartowski: {len(chunks)} paragraphs")
    return {"bartowski": chunks}


def _download_qwen3_dwq() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading mlx-community/qwen3_dwq_calibration_1332 ...")
    ds = load_dataset(
        "mlx-community/qwen3_dwq_calibration_1332", split="train"
    )
    print(f"  loaded {len(ds)} rows")

    chat_texts: list[str] = []
    reasoning_texts: list[str] = []

    for row in ds:
        messages = row["messages"]
        flat = _flatten_sharegpt(messages, include_thinking=True)
        if flat:
            chat_texts.append(flat)

        for msg in messages:
            reasoning = msg.get("reasoning_content")
            content = msg.get("content", "")
            if reasoning:
                r_text = f"<think>\n{reasoning}\n</think>\n{content}"
                reasoning_texts.append(r_text)

    result = {"chat": chat_texts, "reasoning": reasoning_texts}
    for k, v in result.items():
        print(f"  {k}: {len(v)} texts extracted")
    return result


def _download_mixed_exl() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading Orion-zhen/mixed-exl-calibration ...")
    ds = load_dataset("Orion-zhen/mixed-exl-calibration", split="train")
    print(f"  loaded {len(ds)} rows")

    mixed_texts: list[str] = []
    en_texts: list[str] = []
    code_texts: list[str] = []

    for row in ds:
        text = row["content"]
        if not text:
            continue
        mixed_texts.append(text)
        if _has_code(text):
            code_texts.append(text)
        elif _is_english(text):
            en_texts.append(text)

    result = {"mixed": mixed_texts, "en": en_texts, "code": code_texts}
    for k, v in result.items():
        print(f"  {k}: {len(v)} texts extracted")
    return result


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


def _download_firefly_exl() -> dict[str, list[str]]:
    from datasets import load_dataset

    print("Loading Orion-zhen/firefly-exl-calibration (streaming) ...")
    ds = load_dataset(
        "Orion-zhen/firefly-exl-calibration", split="train", streaming=True
    )

    zh_texts: list[str] = []
    pool_size = 2000

    for row in ds.take(pool_size):
        inp = row.get("input", "")
        out = row.get("output", "")
        combined = f"{inp}\n{out}"
        if combined.strip():
            zh_texts.append(combined)

    result = {"zh": zh_texts}
    print(f"  zh: {len(result['zh'])} texts extracted (from {pool_size} scanned)")
    return result


def _download_culturax_ko_ja() -> dict[str, list[str]]:
    from datasets import load_dataset

    result: dict[str, list[str]] = {}
    pool_size = 500

    for lang in ("ko", "ja"):
        print(f"Loading uonlp/CulturaX ({lang}, streaming) ...")
        ds = load_dataset(
            "uonlp/CulturaX", lang, split="train", streaming=True
        )
        texts: list[str] = []
        for row in ds.take(pool_size):
            text = row.get("text", "")
            if text.strip():
                texts.append(text)
        result[lang] = texts
        print(f"  {lang}: {len(texts)} texts extracted (from {pool_size} scanned)")

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

    all_texts: list[str] = []
    pool_size = 1000

    for row in ds.take(pool_size):
        conversations = row.get("conversations", [])
        flat = _flatten_sharegpt(conversations, include_thinking=True)
        if flat:
            all_texts.append(flat)

    result = {"tool_calling": all_texts}
    print(f"  tool_calling (agent traces): {len(all_texts)} texts extracted "
          f"(from {pool_size} scanned)")
    return result


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _merge(target: dict[str, list[str]], source: dict[str, list[str]]) -> None:
    for k, v in source.items():
        target.setdefault(k, []).extend(v)


def build_calibration_data(use_cache: bool = True) -> dict[str, list[str]]:
    sources = [
        ("bartowski", _download_bartowski),
        ("qwen3_dwq", _download_qwen3_dwq),
        ("mixed_exl", _download_mixed_exl),
        ("reasoning_exl", _download_reasoning_exl),
        ("firefly_exl", _download_firefly_exl),
        ("culturax_ko_ja", _download_culturax_ko_ja),
        ("hermes_func", _download_hermes_function_calling),
        ("hermes_agent", _download_hermes_agent_traces),
    ]

    raw: dict[str, list[str]] = {}
    for name, loader in sources:
        _merge(raw, _cached(name, loader, use_cache=use_cache))

    data = {
        "bartowski": raw.get("bartowski", []),
        "chat": _sample(raw.get("chat", []), 500),
        "code": _sample(raw.get("code", []), 500),
        "en": _sample(raw.get("en", []), 400),
        "ja": _sample(raw.get("ja", []), 60),
        "ko": _sample(raw.get("ko", []), 60),
        "mixed": _sample(raw.get("mixed", []), 800),
        "reasoning": _sample(raw.get("reasoning", []), 500),
        "tool_calling": _sample(raw.get("tool_calling", []), 500),
        "zh": _sample(raw.get("zh", []), 400),
    }

    print("\n--- Final counts ---")
    total = 0
    for k in sorted(data.keys()):
        print(f"  {k}: {len(data[k])}")
        total += len(data[k])
    print(f"  TOTAL: {total}")
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
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

    random.seed(args.seed)
    data = build_calibration_data(use_cache=not args.no_cache)

    total = sum(len(v) for v in data.values())
    size_kb = sum(len(t) for v in data.values() for t in v) / 1024
    print(f"\n{total} samples, {size_kb:.0f} KB text")

    if args.dry_run:
        print("Dry run — not writing file.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Written to {out} ({out.stat().st_size / 1024:.0f} KB on disk)")


if __name__ == "__main__":
    main()
