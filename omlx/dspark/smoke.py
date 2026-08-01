"""Startup correctness probe for verified cross-target pairs."""

from __future__ import annotations

from typing import Any


def verify_cross_target_smoke(provider: Any, tokenizer: Any) -> None:
    prompt_ids = [int(token) for token in tokenizer.encode("dSpark smoke test: 1+1=")]
    try:
        provider.greedy_smoke(prompt_ids, max_tokens=4)
    except Exception as exc:
        raise ValueError(f"verified_cross_target smoke test failed: {exc}") from exc
