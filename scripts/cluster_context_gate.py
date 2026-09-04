#!/usr/bin/env python3
"""Run an exact-token streaming context gate against an oMLX endpoint.

This is intentionally a small black-box harness: it builds a prompt whose
token count is verified with the model tokenizer, sends it through the public
OpenAI-compatible API, and writes one JSON result that survives the invoking
terminal.  It is useful for long hardware gates where keeping pytest or a
browser request open would make the coordinator's lifetime part of the test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--completion-tokens", type=int, default=2)
    parser.add_argument("--read-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("~/.omlx/settings.json").expanduser(),
    )
    return parser.parse_args()


def _api_key(path: Path) -> str:
    environment_key = os.environ.get("OMLX_API_KEY", "").strip()
    if environment_key:
        return environment_key
    settings = json.loads(path.read_text())
    key = str(settings.get("auth", {}).get("api_key", "")).strip()
    if not key:
        raise RuntimeError(f"no API key in {path}")
    return key


def _exact_prompt(tokenizer_path: Path, target: int) -> str:
    if target < 1:
        raise ValueError("prompt token count must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=False,
    )
    unit = " hello"
    unit_tokens = tokenizer.encode(unit, add_special_tokens=False)
    if len(unit_tokens) != 1:
        raise RuntimeError(
            f"gate prompt unit encoded to {len(unit_tokens)} tokens, expected 1"
        )
    prompt = unit * target
    measured = len(tokenizer.encode(prompt, add_special_tokens=False))
    if measured != target:
        raise RuntimeError(
            f"gate prompt encoded to {measured} tokens, expected {target}"
        )
    return prompt


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    if args.read_timeout_seconds <= 0:
        raise ValueError("read timeout must be positive")
    started = time.monotonic()
    result: dict[str, Any] = {
        "model": args.model,
        "prompt_tokens_requested": args.prompt_tokens,
        "completion_tokens_requested": args.completion_tokens,
        "status": "running",
    }
    _write_result(args.output, result)
    try:
        prompt_started = time.monotonic()
        prompt = _exact_prompt(args.tokenizer, args.prompt_tokens)
        result["prompt_build_seconds"] = time.monotonic() - prompt_started
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.completion_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first_token_at: float | None = None
        completion = ""
        usage: dict[str, Any] = {}
        # MLX-LM emits SSE keepalives during a long prefill. This is therefore
        # an inactivity bound, not a total 256K deadline: an advancing request
        # can run for hours, while a dead collective cannot hang the gate
        # forever.
        timeout = httpx.Timeout(
            connect=10.0,
            read=args.read_timeout_seconds,
            write=60.0,
            pool=10.0,
        )
        with httpx.Client(timeout=timeout) as client, client.stream(
            "POST",
            f"{args.base_url.rstrip('/')}/v1/completions",
            headers={"Authorization": f"Bearer {_api_key(args.api_key_file)}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                text = "".join(
                    str(choice.get("text") or "")
                    for choice in event.get("choices", ())
                )
                if text and first_token_at is None:
                    first_token_at = time.monotonic()
                completion += text
                if event.get("usage"):
                    usage = event["usage"]
        finished = time.monotonic()
        result.update(
            status="passed",
            elapsed_seconds=finished - started,
            time_to_first_token_seconds=(
                first_token_at - started if first_token_at is not None else None
            ),
            completion=completion,
            usage=usage,
        )
        if int(usage.get("prompt_tokens", -1)) != args.prompt_tokens:
            raise RuntimeError(
                "server usage did not confirm the requested prompt length: "
                f"{usage.get('prompt_tokens')!r}"
            )
        if int(usage.get("completion_tokens", 0)) < 1:
            raise RuntimeError("server returned no completion token")
    except BaseException as exc:
        result.update(
            status="failed",
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_result(args.output, result)
        raise
    _write_result(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
