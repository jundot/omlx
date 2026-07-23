#!/usr/bin/env python3
"""Replay a captured Anthropic /v1/messages payload against an oMLX server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DSML_MARKERS = (
    "<｜DSML｜tool_calls>",
    "</｜DSML｜tool_calls>",
    "<｜DSML｜invoke",
    "</｜DSML｜invoke>",
    "<｜DSML｜parameter",
    "</｜DSML｜parameter>",
)


def _apply_overrides(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.model is not None:
        payload["model"] = args.model
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.stream is not None:
        payload["stream"] = args.stream


def _extract_text_from_event(event: dict[str, Any]) -> str:
    pieces: list[str] = []
    delta = event.get("delta")
    if isinstance(delta, dict):
        text = delta.get("text") or delta.get("thinking")
        if isinstance(text, str):
            pieces.append(text)
    block = event.get("content_block")
    if isinstance(block, dict):
        text = block.get("text") or block.get("thinking")
        if isinstance(text, str):
            pieces.append(text)
    content = event.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "".join(pieces)


def _first_marker_excerpt(text: str, radius: int = 160) -> str:
    positions = [text.find(marker) for marker in DSML_MARKERS if marker in text]
    if not positions:
        return ""
    pos = min(positions)
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return text[start:end].replace("\n", "\\n")


def _summarize(
    name: str, raw: str, text: str, elapsed: float, error: str = ""
) -> dict[str, Any]:
    combined = text or raw
    return {
        "name": name,
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "text_chars": len(text),
        "raw_chars": len(raw),
        "dsml_marker_count": sum(combined.count(marker) for marker in DSML_MARKERS),
        "orphan_command_param": '<｜DSML｜parameter name="command" string="true"></｜DSML｜parameter>'
        in combined,
        "first_marker_excerpt": _first_marker_excerpt(combined),
    }


def _post_once(
    *,
    name: str,
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    output_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if payload.get("stream") else "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    raw_parts: list[str] = []
    text_parts: list[str] = []
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                raw_parts.append(line)
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text_parts.append(_extract_text_from_event(event))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        elapsed = time.monotonic() - started
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
        return _summarize(name, raw, "", elapsed, error=f"HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        elapsed = time.monotonic() - started
        return _summarize(
            name, "".join(raw_parts), "".join(text_parts), elapsed, error=repr(exc)
        )

    elapsed = time.monotonic() - started
    raw = "".join(raw_parts)
    text = "".join(text_parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
    (output_dir / f"{name}.text.txt").write_text(text, encoding="utf-8")
    return _summarize(name, raw, text, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18150/v1/messages")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--output-dir", default=Path("/tmp/omlx_repro_outputs"), type=Path
    )
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--concurrency", default=1, type=int)
    parser.add_argument("--stagger-s", default=0.0, type=float)
    parser.add_argument("--timeout-s", default=1800.0, type=float)
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    _apply_overrides(payload, args)

    print(
        json.dumps(
            {
                "url": args.url,
                "payload": str(args.payload),
                "model": payload.get("model"),
                "messages": len(payload.get("messages") or []),
                "tools": len(payload.get("tools") or []),
                "max_tokens": payload.get("max_tokens"),
                "temperature": payload.get("temperature", "<server-default>"),
                "top_p": payload.get("top_p", "<server-default>"),
                "stream": payload.get("stream"),
                "repeat": args.repeat,
                "concurrency": args.concurrency,
            },
            ensure_ascii=False,
        )
    )

    def run(index: int) -> dict[str, Any]:
        if args.stagger_s > 0:
            time.sleep(args.stagger_s * index)
        return _post_once(
            name=f"run_{index:03d}",
            url=args.url,
            payload=dict(payload),
            api_key=args.api_key,
            output_dir=args.output_dir,
            timeout=args.timeout_s,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [executor.submit(run, i) for i in range(args.repeat)]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
