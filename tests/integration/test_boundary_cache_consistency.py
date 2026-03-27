# SPDX-License-Identifier: Apache-2.0
"""
Boundary cache consistency tests for all cache types.

Verifies that boundary caching ON/OFF and SSD cache hit/miss produce
identical token-level outputs at temperature=0 across:
- KVCache only (MiniMax-M2.5)
- ArraysCache hybrid non-MoE (Qwen3.5-27B)
- ArraysCache hybrid MoE (Qwen3.5-35B-A3B)
- RotatingKVCache + KVCache hybrid (gpt-oss-120b)

For RotatingKVCache models, boundary ON/OFF may produce different tokens
due to chunk size differences (no recurrent state accumulation, so this
is a precision difference, not degradation). These models only check
output quality and SSD cache hit consistency.

Run with: pytest tests/integration/test_boundary_cache_consistency.py -v -m slow -s
"""

import gc
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Requires macOS with Apple Silicon",
    ),
]

MODELS = {
    "kvcache": {
        "path": "/Users/cryingneko/Workspace/models/MiniMax-M2.5-oQ3e",
        "desc": "KVCache only (MiniMax-M2.5)",
        "expect_on_off_match": True,
    },
    "arrayscache_dense": {
        "path": "/Users/cryingneko/Workspace/models/Qwen3.5-27B-8bit",
        "desc": "ArraysCache hybrid non-MoE (Qwen3.5-27B)",
        "expect_on_off_match": True,
    },
    "arrayscache_moe": {
        "path": "/Users/cryingneko/Workspace/models/Qwen3.5-35B-A3B-oQ4",
        "desc": "ArraysCache hybrid MoE (Qwen3.5-35B-A3B)",
        "expect_on_off_match": True,
    },
    "rotating_hybrid": {
        "path": "/Volumes/SSD/Models/gpt-oss-120b-MXFP4-Q8",
        "desc": "RotatingKVCache+KVCache hybrid (gpt-oss-120b)",
        "expect_on_off_match": False,  # chunk size differs, quality-only check
    },
    "rotating_vlm": {
        "path": "/Users/cryingneko/Workspace/models/gemma-3-12b-it-qat-4bit",
        "desc": "RotatingKVCache+KVCache VLM hybrid (Gemma3-12B-QAT)",
        "expect_on_off_match": False,  # chunk size differs for RotatingKVCache
    },
}


def _build_8k_prompt(tokenizer) -> List[int]:
    """Build a prompt of ~8K tokens using chat template."""
    base_text = (
        "You are an expert software engineer. "
        "You have deep knowledge of Python, Rust, C++, and JavaScript. "
        "You follow best practices and write clean, maintainable code. "
        "You always consider edge cases and error handling. "
        "You write comprehensive tests for all your code. "
    )
    long_system = base_text * 80

    question = (
        "Explain the difference between a stack and a queue. "
        "Give examples in Python with type hints."
    )

    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": question},
    ]

    try:
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    except Exception:
        text = f"{long_system}\n\nUser: {question}\n\nAssistant:"
        token_ids = tokenizer.encode(text)

    target = 8192
    if len(token_ids) > target:
        token_ids = token_ids[:target]
    elif len(token_ids) < target - 500:
        extra = base_text * 30
        messages[0]["content"] += extra
        try:
            token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            text = f"{messages[0]['content']}\n\nUser: {question}\n\nAssistant:"
            token_ids = tokenizer.encode(text)
        if len(token_ids) > target:
            token_ids = token_ids[:target]

    return token_ids


def _generate_tokens(
    model,
    tokenizer,
    prompt_token_ids: List[int],
    *,
    max_tokens: int = 100,
    ssd_cache_dir: Optional[str] = None,
    block_size: int = 2048,
) -> Tuple[List[int], int]:
    """Run generation and return (output_token_ids, cached_tokens)."""
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    config_kwargs = dict(
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        completion_batch_size=1,
        prefill_step_size=2048,
    )

    if ssd_cache_dir is not None:
        config_kwargs["paged_ssd_cache_dir"] = ssd_cache_dir
        config_kwargs["paged_cache_block_size"] = block_size
        config_kwargs["paged_ssd_cache_max_size"] = 10 * 1024 * 1024 * 1024

    config = SchedulerConfig(**config_kwargs)
    scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer)

    request = Request(
        request_id="test",
        prompt=prompt_token_ids,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        ),
    )

    scheduler.add_request(request)

    cached_tokens = 0
    output_token_ids = []

    for _ in range(max_tokens + 200):
        step_result = scheduler.step()

        for output in step_result.outputs:
            if output.cached_tokens > 0:
                cached_tokens = output.cached_tokens
            if output.finished:
                output_token_ids = list(output.output_token_ids)
                break

        if step_result.finished_request_ids:
            break

    scheduler.shutdown()
    return output_token_ids, cached_tokens


def _check_output_quality(text: str, model_desc: str):
    """Check that output is coherent, not gibberish."""
    assert len(text.strip()) > 0, f"[{model_desc}] Empty output"

    words = text.split()
    assert len(words) >= 5, (
        f"[{model_desc}] Too few words ({len(words)}): {text!r}"
    )

    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / max(len(text), 1)
    assert alpha_ratio > 0.3, (
        f"[{model_desc}] Low alpha ratio ({alpha_ratio:.2f}), "
        f"possibly gibberish: {text[:200]!r}"
    )

    for i in range(len(text) - 20):
        if len(set(text[i : i + 20])) == 1:
            pytest.fail(
                f"[{model_desc}] Excessive single-char repetition: "
                f"{text[max(0,i-5):i+25]!r}"
            )


def _run_model_test(model_path: str, model_desc: str, expect_on_off_match: bool):
    """Run full boundary cache consistency test for a single model."""
    import mlx.core as mx
    from mlx_lm import load

    from omlx.patches.gated_delta_advance import (
        apply_gated_delta_advance_patch,
    )

    print(f"\n{'='*60}")
    print(f"Testing: {model_desc}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    model, tokenizer = load(model_path)
    patch_applied = apply_gated_delta_advance_patch(model)
    print(f"  GatedDeltaNet advance patch: {'applied' if patch_applied else 'skipped (not needed)'}")

    prompt_token_ids = _build_8k_prompt(tokenizer)
    print(f"  Prompt tokens: {len(prompt_token_ids)}")

    # --- Test 1: Boundary ON vs OFF ---
    print("\n  [Test 1] Boundary cache ON vs OFF...")

    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_")
    try:
        tokens_on, _ = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    tokens_off, _ = _generate_tokens(
        model, tokenizer, prompt_token_ids,
        ssd_cache_dir=None,
    )

    text_on = tokenizer.decode(tokens_on)
    text_off = tokenizer.decode(tokens_off)

    print(f"    ON  ({len(tokens_on)} tokens): {text_on[:120]}...")
    print(f"    OFF ({len(tokens_off)} tokens): {text_off[:120]}...")

    _check_output_quality(text_on, f"{model_desc} boundary-ON")
    _check_output_quality(text_off, f"{model_desc} boundary-OFF")
    print("    Quality check: PASSED")

    match = tokens_on == tokens_off
    if match:
        print("    Token match: IDENTICAL ✓")
    else:
        min_len = min(len(tokens_on), len(tokens_off))
        diff_idx = next(
            (i for i in range(min_len) if tokens_on[i] != tokens_off[i]),
            min_len,
        )
        print(f"    Token match: DIFFER at position {diff_idx}")
        print(f"      ON[{diff_idx}]: {tokens_on[diff_idx] if diff_idx < len(tokens_on) else 'END'}")
        print(f"      OFF[{diff_idx}]: {tokens_off[diff_idx] if diff_idx < len(tokens_off) else 'END'}")

    if expect_on_off_match:
        assert match, f"[{model_desc}] Boundary ON/OFF tokens differ"
    else:
        if not match:
            print(
                "    (Expected: RotatingKVCache chunk size differs from "
                "prefill_step_size — no recurrent state, quality OK)"
            )

    # --- Test 2: SSD cache hit vs fresh ---
    print("\n  [Test 2] SSD cache hit vs fresh prefill...")

    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_ssd_")
    try:
        tokens_fresh, cached_fresh = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
        )
        print(f"    Fresh: {len(tokens_fresh)} tokens, cached={cached_fresh}")

        tokens_cached, cached_count = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
        )
        print(f"    Cached: {len(tokens_cached)} tokens, cached={cached_count}")

        text_cached = tokenizer.decode(tokens_cached)
        _check_output_quality(text_cached, f"{model_desc} cached")
        print("    Quality check: PASSED")

        match_ssd = tokens_fresh == tokens_cached
        if match_ssd:
            print("    Token match: IDENTICAL ✓")
        else:
            min_len = min(len(tokens_fresh), len(tokens_cached))
            diff_idx = next(
                (i for i in range(min_len) if tokens_fresh[i] != tokens_cached[i]),
                min_len,
            )
            print(f"    Token match: DIFFER at position {diff_idx}")

        if cached_count > 0:
            print(f"    Cache hit confirmed: {cached_count} tokens from SSD ✓")
        else:
            print("    WARNING: No cache hit detected")

        assert match_ssd, f"[{model_desc}] SSD cache hit/fresh tokens differ"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  [{model_desc}] ALL TESTS PASSED ✓")

    del model, tokenizer
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize(
    "model_key",
    list(MODELS.keys()),
    ids=[m["desc"] for m in MODELS.values()],
)
def test_boundary_cache_consistency(model_key):
    """Test boundary cache consistency for each model type."""
    info = MODELS[model_key]
    if not Path(info["path"]).exists():
        pytest.skip(f"Model not found: {info['path']}")
    _run_model_test(info["path"], info["desc"], info["expect_on_off_match"])
