#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001
"""CPU-preparable restore QA for the synthetic Qwen3.8 16K cache case.

The cached phase primes the server, then checks three saved continuation
fixtures for the expected recurrent-checkpoint walkback. The reference phase
sends the same fixture bytes to a fresh no-cache server and requires zero
cached tokens. ``--create-only`` performs local tokenization and fixture
writes without HTTP or GPU work.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from bench_qwen35_cache import (  # noqa: E402
    FACT,
    MODEL,
    case_messages,
    prompt_metadata,
    request_stream,
    validate_base_url,
)


DEFAULT_TOKENIZER = (
    Path.home() / ".mtplx/models/Qwen--Qwen3.8-27B-oQ4e-mtp/tokenizer.json"
)
DEFAULT_FIXTURES = REPOSITORY_ROOT / "results/qwen35-restore-fixtures.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/qwen35-restore-probe.jsonl"


def _solid_red_png(width: int = 64, height: int = 64) -> bytes:
    """Build a 64x64 RGB PNG without external image dependencies."""

    import struct

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        checksum = zlib.crc32(body) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", checksum)
        )

    return (
        signature
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def _vision_messages() -> list[dict[str, Any]]:
    png = base64.b64encode(_solid_red_png()).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What colour is the square? Answer one word.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png}"},
                },
            ],
        }
    ]


def _load_tokenizer(path: Path) -> Tokenizer:
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {path}")
    return Tokenizer.from_file(str(path))


def _fixture_messages(
    tokenizer: Tokenizer,
    raw: str,
    cut: int,
    suffix: str,
) -> tuple[list[dict[str, str]], int]:
    ids = tokenizer.encode(raw).ids
    prefix = tokenizer.decode(ids[:cut], skip_special_tokens=False)
    content = prefix + suffix
    return (
        [{"role": "user", "content": content}],
        len(tokenizer.encode(content).ids),
    )


def build_fixtures(tokenizer: Tokenizer, seed: int) -> dict[str, Any]:
    prime, max_tokens, prime_meta = case_messages(
        "context16k-retrieval", seed, None
    )
    raw = str(prime[0]["content"])
    cuts = (
        (
            3500,
            "cobalt-heron-7913",
            2048,
            "\n\nNew continuation fact: PLANTED_FACT_3070 is "
            "cobalt-heron-7913. What is the exact new value? /no_think",
        ),
        (
            5800,
            "amber-otter-4821",
            4096,
            "\n\nContinue and answer the original PLANTED_FACT_3070 value only. "
            "/no_think",
        ),
        (
            8300,
            "amber-otter-4821",
            8192,
            "\n\nAnswer the original PLANTED_FACT_3070 value only. /no_think",
        ),
    )
    cases: dict[str, Any] = {}
    for cut, expected, walkback, suffix in cuts:
        messages, token_count = _fixture_messages(tokenizer, raw, cut, suffix)
        cases[f"cut{cut}"] = {
            "case": f"cut{cut}",
            "cut_tokens": cut,
            "messages": messages,
            "token_count": token_count,
            "expected_answer": expected,
            "expected_cache_tokens": walkback,
            "max_tokens": 32,
            "seed": seed + cut,
            "meta": prompt_metadata(
                messages,
                {"kind": "restore_probe", "cut_tokens": cut},
            ),
        }
    return {
        "schema": 1,
        "synthetic": True,
        "model": MODEL,
        "tokenizer": "Qwen3.8 tokenizer.json",
        "prime": {
            "messages": prime,
            "max_tokens": max_tokens,
            "seed": seed,
            "token_count": len(tokenizer.encode(raw).ids),
            "meta": prompt_metadata(prime, prime_meta),
        },
        "cases": cases,
    }


def _write_fixture(path: Path, fixture: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_fixture(fixture: dict[str, Any]) -> None:
    """Verify that the cuts still exercise the intended fact cases."""

    cases = fixture.get("cases", {})
    for name, item in cases.items():
        content = str((item.get("messages") or [{}])[0].get("content", ""))
        if name == "cut3500" and "amber-otter-4821" in content:
            raise ValueError("cut3500 unexpectedly retains the original fact")
        if name in {"cut5800", "cut8300"} and FACT not in content:
            raise ValueError(f"{name} does not retain the original planted fact")


def _normalise_marker(value: Any) -> str:
    """Ignore only whitespace and punctuation around an answer marker."""

    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _record_request(
    *,
    base_url: str,
    model: str,
    phase: str,
    case: str,
    item: dict[str, Any],
    log_path: Path | None,
    timeout: float,
) -> dict[str, Any]:
    started = time.time()
    metrics = request_stream(
        base_url,
        model,
        item["messages"],
        item["max_tokens"],
        item["seed"],
        timeout,
        log_path,
    )
    output = metrics.get("output") or metrics.get("content") or ""
    expected_cache = item.get("expected_cache_tokens")
    actual_cache = metrics.get("cache_tokens")
    is_prime = case == "prime"
    stream_ok = (
        metrics.get("stream_valid") is True
        and metrics.get("stream_done") is True
        and not metrics.get("error")
        and not metrics.get("stream_error")
    )
    checks = {
        "answer_ok": _normalise_marker(item["expected_answer"])
        in _normalise_marker(output),
        "stream_ok": stream_ok,
        "cache_ok": (
            True
            if is_prime
            else actual_cache == 0
            if phase == "reference"
            else actual_cache == expected_cache
        ),
    }
    return {
        "schema": 1,
        "synthetic": True,
        "model": model,
        "phase": phase,
        "case": case,
        "started_unix": started,
        "expected_cache_tokens": expected_cache,
        "actual_cache_tokens": actual_cache,
        "ttft_s": metrics.get("ttft_s"),
        "end_to_end_s": metrics.get("end_to_end_s"),
        "usage": metrics.get("usage_raw"),
        "output": output,
        "checks": checks,
        "metrics": metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Explicit isolated server URL, normally http://127.0.0.1:8845",
    )
    parser.add_argument("--allow-8843", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("cached", "reference"),
        default="cached",
    )
    parser.add_argument("--model", default=MODEL, help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--serve-log", type=Path)
    parser.add_argument("--seed", type=int, default=3070)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Run the optional generated red-square vision request",
    )
    args = parser.parse_args(argv)
    try:
        args.base_url = validate_base_url(args.base_url, args.allow_8843)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fixture_path = args.fixtures.expanduser()
    if fixture_path.exists() and not args.create_only:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        if fixture.get("synthetic") is not True:
            raise ValueError(f"fixture is not marked synthetic: {fixture_path}")
    else:
        tokenizer = _load_tokenizer(args.tokenizer.expanduser())
        fixture = build_fixtures(tokenizer, args.seed)
        _write_fixture(fixture_path, fixture)
    _validate_fixture(fixture)
    if args.create_only:
        print(
            json.dumps(
                {"created": str(fixture_path), "synthetic": True},
                indent=2,
            )
        )
        return 0

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.serve_log.expanduser() if args.serve_log else None
    failures = 0
    with output_path.open("a", encoding="utf-8") as output:
        if args.phase == "cached":
            prime = fixture["prime"]
            prime_row = _record_request(
                base_url=args.base_url,
                model=args.model,
                phase=args.phase,
                case="prime",
                item={
                    **prime,
                    "expected_answer": "amber-otter-4821",
                    "expected_cache_tokens": 0,
                },
                log_path=log_path,
                timeout=args.timeout,
            )
            output.write(json.dumps(prime_row, ensure_ascii=False) + "\n")
            failures += sum(not value for value in prime_row["checks"].values())
        for case, item in fixture["cases"].items():
            row = _record_request(
                base_url=args.base_url,
                model=args.model,
                phase=args.phase,
                case=case,
                item=item,
                log_path=log_path,
                timeout=args.timeout,
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            failures += sum(not value for value in row["checks"].values())
        if args.vision:
            item = {
                "messages": _vision_messages(),
                "max_tokens": 8,
                "seed": args.seed + 9000,
                "expected_answer": "red",
                "expected_cache_tokens": 0,
            }
            row = _record_request(
                base_url=args.base_url,
                model=args.model,
                phase=args.phase,
                case="vision-red-square",
                item=item,
                log_path=log_path,
                timeout=args.timeout,
            )
            row["checks"]["vision_answer_ok"] = row["checks"]["answer_ok"]
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            failures += sum(not value for value in row["checks"].values())
        output.flush()
    if failures:
        print(f"restore probe failed checks={failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
