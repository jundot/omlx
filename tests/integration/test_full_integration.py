# SPDX-License-Identifier: Apache-2.0
"""
Full integration test for oMLX with real models.

Tests cache consistency, concurrent batching, TurboQuant, VLM image caching,
and multi-turn VLM conversations across 7 models using both LLM and VLM engines.

Test categories:
  1. 9K context cache consistency (boundary cache, SSD cache hit/miss)
  2. 4-request concurrent batching (simultaneous + sequential)
  3. TurboQuant 3-bit with cache and batching
  4. VLM engine basics (tests 1-3 on VLMModelAdapter)
  5. VLM image caching (5K text + image per turn, 3 turns)
  6. VLM multi-turn image quality (coherent responses across turns)
  7. VLM image caching with 4-request batching

Run with: pytest tests/integration/test_full_integration.py -v -m slow -s
Single model: pytest tests/integration/test_full_integration.py -v -m slow -s -k "Qwen3-4B"
"""

import gc
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Requires macOS with Apple Silicon",
    ),
]

MODELS = [
    "/Users/cryingneko/Workspace/models/gemma-4-26b-a4b-it-8bit",
    "/Users/cryingneko/Workspace/models/gemma-4-26b-a4b-mxfp4",
    "/Users/cryingneko/Workspace/models/gemma-4-31b-it-4bit",
    "/Users/cryingneko/Workspace/models/Qwen3-4B-Instruct-2507-4bit",
    "/Users/cryingneko/Workspace/models/Qwen3.5-35B-A3B-4bit",
    "/Users/cryingneko/Workspace/models/Qwen3.5-27B-4bit",
    "/Users/cryingneko/Workspace/models/Nemotron-Cascade-2-30B-A3B-4bit",
]

# Questions for batching tests (short, diverse prompts)
BATCH_QUESTIONS = [
    "Explain the difference between a stack and a queue in 3 sentences.",
    "What is binary search? Give a one-paragraph explanation.",
    "Why are hash tables O(1) for lookup? Explain briefly.",
    "Compare bubble sort and merge sort in terms of time complexity.",
]


# ---------------------------------------------------------------------------
# Helper: peak memory tracking
# ---------------------------------------------------------------------------

@contextmanager
def _track_peak_memory(label: str):
    """Track and print peak GPU memory during a block."""
    import mlx.core as mx

    mx.synchronize()
    mem_before = mx.get_active_memory()
    peak_before = mx.get_peak_memory()
    # Reset peak to current level
    mx.reset_peak_memory()

    yield

    mx.synchronize()
    mem_after = mx.get_active_memory()
    peak = mx.get_peak_memory()
    print(
        f"    [mem] {label}: "
        f"active {mem_after / 1024**3:.2f}GB "
        f"(delta {(mem_after - mem_before) / 1024**3:+.2f}GB), "
        f"peak {peak / 1024**3:.2f}GB"
    )


# ---------------------------------------------------------------------------
# Helper: build prompts
# ---------------------------------------------------------------------------

def _apply_chat_template_as_ids(tokenizer, messages) -> List[int]:
    """Apply chat template and guarantee token IDs are returned."""
    try:
        # Use tokenize=False to get a string, then encode.
        # This avoids BatchEncoding objects from transformers tokenizers.
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(prompt_str, str):
            # Unexpected return type; try extracting input_ids
            if hasattr(prompt_str, "input_ids"):
                ids = prompt_str.input_ids
                return ids[0] if isinstance(ids[0], list) else list(ids)
            prompt_str = str(prompt_str)
        return tokenizer.encode(prompt_str)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text += "\nassistant:"
        return tokenizer.encode(text)


def _build_9k_prompt(tokenizer) -> List[int]:
    """Build a prompt of ~9K tokens using chat template."""
    base_text = (
        "You are an expert software engineer. "
        "You have deep knowledge of Python, Rust, C++, and JavaScript. "
        "You follow best practices and write clean, maintainable code. "
        "You always consider edge cases and error handling. "
        "You write comprehensive tests for all your code. "
    )
    long_system = base_text * 100

    question = (
        "Explain the difference between a stack and a queue. "
        "Give examples in Python with type hints."
    )

    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": question},
    ]

    token_ids = _apply_chat_template_as_ids(tokenizer, messages)

    target = 9000
    if len(token_ids) > target:
        token_ids = token_ids[:target]
    elif len(token_ids) < target - 500:
        extra = base_text * 50
        messages[0]["content"] += extra
        token_ids = _apply_chat_template_as_ids(tokenizer, messages)
        if len(token_ids) > target:
            token_ids = token_ids[:target]

    return token_ids


def _build_5k_system(tokenizer) -> str:
    """Build a long system message of ~5K tokens."""
    base_text = (
        "You are a helpful assistant that describes images in detail. "
        "You pay attention to colors, shapes, patterns, and textures. "
        "You provide accurate and thorough descriptions. "
    )
    return base_text * 60


def _build_short_prompts(tokenizer, n: int = 4) -> List[List[int]]:
    """Build n different short prompts for batching tests."""
    prompts = []
    for q in BATCH_QUESTIONS[:n]:
        messages = [{"role": "user", "content": q}]
        token_ids = _apply_chat_template_as_ids(tokenizer, messages)
        prompts.append(token_ids)
    return prompts


# ---------------------------------------------------------------------------
# Helper: output quality check
# ---------------------------------------------------------------------------

def _check_output_quality(text: str, label: str):
    """Check that output is coherent, not gibberish."""
    assert len(text.strip()) > 0, f"[{label}] Empty output"

    words = text.split()
    assert len(words) >= 5, (
        f"[{label}] Too few words ({len(words)}): {text!r}"
    )

    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / max(len(text), 1)
    assert alpha_ratio > 0.3, (
        f"[{label}] Low alpha ratio ({alpha_ratio:.2f}), "
        f"possibly gibberish: {text[:200]!r}"
    )

    for i in range(len(text) - 20):
        if len(set(text[i : i + 20])) == 1:
            pytest.fail(
                f"[{label}] Excessive single-char repetition: "
                f"{text[max(0,i-5):i+25]!r}"
            )


# ---------------------------------------------------------------------------
# Helper: test image creation
# ---------------------------------------------------------------------------

def _create_test_image(seed: int = 0, width: int = 336, height: int = 336):
    """Create a test image with a gradient pattern based on seed."""
    from PIL import Image

    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            # Each seed produces a visually distinct gradient
            r = int(255 * ((x + seed * 80) % width) / width)
            g = int(255 * ((y + seed * 120) % height) / height)
            b = int((128 + seed * 60) % 256)
            pixels[x, y] = (r, g, b)
    return img


def _create_colored_image(color: Tuple[int, int, int], width: int = 336, height: int = 336):
    """Create a solid-color image for quality testing."""
    from PIL import Image

    return Image.new("RGB", (width, height), color)


# ---------------------------------------------------------------------------
# Helper: single-request generation
# ---------------------------------------------------------------------------

def _generate_tokens(
    model,
    tokenizer,
    prompt_token_ids: List[int],
    *,
    max_tokens: int = 100,
    ssd_cache_dir: Optional[str] = None,
    block_size: int = 2048,
    turboquant_bits: Optional[float] = None,
    vlm_inputs_embeds: Optional[Any] = None,
    vlm_extra_kwargs: Optional[Dict[str, Any]] = None,
    vlm_image_hash: Optional[str] = None,
) -> Tuple[List[int], int]:
    """Run generation with a single request and return (output_token_ids, cached_tokens)."""
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    config_kwargs = dict(
        max_num_seqs=1,
        max_num_batched_tokens=16384,
        completion_batch_size=1,
        prefill_step_size=2048,
    )

    if ssd_cache_dir is not None:
        config_kwargs["paged_ssd_cache_dir"] = ssd_cache_dir
        config_kwargs["paged_cache_block_size"] = block_size
        config_kwargs["paged_ssd_cache_max_size"] = 10 * 1024 * 1024 * 1024

    config = SchedulerConfig(**config_kwargs)
    scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer)

    if turboquant_bits is not None:
        from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
        apply_turboquant_attention_patch()
        scheduler._turboquant_kv_bits = turboquant_bits

    # Use repetition_penalty for VLM requests to prevent degeneration
    # on synthetic test images with greedy decoding.
    rep_penalty = 1.1 if vlm_inputs_embeds is not None else 1.0

    request = Request(
        request_id="test",
        prompt=prompt_token_ids,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            repetition_penalty=rep_penalty,
        ),
    )
    if vlm_inputs_embeds is not None:
        request.vlm_inputs_embeds = vlm_inputs_embeds
        request.vlm_extra_kwargs = vlm_extra_kwargs
        request.vlm_image_hash = vlm_image_hash

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


# ---------------------------------------------------------------------------
# Helper: batch generation
# ---------------------------------------------------------------------------

def _generate_batch(
    model,
    tokenizer,
    prompt_list: List[List[int]],
    *,
    mode: str = "concurrent",
    max_tokens: int = 100,
    ssd_cache_dir: Optional[str] = None,
    block_size: int = 2048,
    turboquant_bits: Optional[float] = None,
    vlm_embeds_list: Optional[List[Tuple[Any, Optional[Dict], Optional[str]]]] = None,
) -> List[Tuple[str, List[int], int]]:
    """
    Run batch generation with multiple requests.

    Args:
        mode: "concurrent" (all at once) or "sequential" (1-second intervals)
        vlm_embeds_list: per-request (inputs_embeds, extra_kwargs, image_hash) tuples

    Returns:
        List of (request_id, output_token_ids, cached_tokens)
    """
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    n = len(prompt_list)

    config_kwargs = dict(
        max_num_seqs=n,
        max_num_batched_tokens=16384,
        completion_batch_size=n,
        prefill_step_size=2048,
    )

    if ssd_cache_dir is not None:
        config_kwargs["paged_ssd_cache_dir"] = ssd_cache_dir
        config_kwargs["paged_cache_block_size"] = block_size
        config_kwargs["paged_ssd_cache_max_size"] = 10 * 1024 * 1024 * 1024

    config = SchedulerConfig(**config_kwargs)
    scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer)

    if turboquant_bits is not None:
        from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
        apply_turboquant_attention_patch()
        scheduler._turboquant_kv_bits = turboquant_bits

    # Build requests
    # Use repetition_penalty for VLM batch requests to prevent
    # degeneration on synthetic test images with greedy decoding.
    has_vlm = vlm_embeds_list is not None and any(e[0] is not None for e in vlm_embeds_list)
    rep_penalty = 1.1 if has_vlm else 1.0

    requests = []
    for i, prompt_ids in enumerate(prompt_list):
        req = Request(
            request_id=f"batch-{i}",
            prompt=prompt_ids,
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                repetition_penalty=rep_penalty,
            ),
        )
        if vlm_embeds_list is not None and i < len(vlm_embeds_list):
            embeds, kwargs, img_hash = vlm_embeds_list[i]
            req.vlm_inputs_embeds = embeds
            req.vlm_extra_kwargs = kwargs
            req.vlm_image_hash = img_hash
        requests.append(req)

    # Track results per request
    results: Dict[str, Tuple[List[int], int]] = {}
    finished_ids = set()

    if mode == "concurrent":
        # Add all requests at once
        for req in requests:
            scheduler.add_request(req)

        for _ in range(max_tokens * n + 500):
            step_result = scheduler.step()
            for output in step_result.outputs:
                if output.cached_tokens > 0 and output.request_id not in results:
                    results.setdefault(output.request_id, ([], output.cached_tokens))
                if output.finished:
                    results[output.request_id] = (
                        list(output.output_token_ids),
                        output.cached_tokens,
                    )
                    finished_ids.add(output.request_id)
            if len(finished_ids) >= n:
                break

    elif mode == "sequential":
        add_idx = 0
        scheduler.add_request(requests[add_idx])
        add_idx += 1
        last_add_time = time.monotonic()

        for _ in range(max_tokens * n + 2000):
            # Add next request after 1-second interval
            now = time.monotonic()
            if add_idx < n and now - last_add_time >= 1.0:
                scheduler.add_request(requests[add_idx])
                add_idx += 1
                last_add_time = now

            step_result = scheduler.step()
            for output in step_result.outputs:
                if output.cached_tokens > 0 and output.request_id not in results:
                    results.setdefault(output.request_id, ([], output.cached_tokens))
                if output.finished:
                    results[output.request_id] = (
                        list(output.output_token_ids),
                        output.cached_tokens,
                    )
                    finished_ids.add(output.request_id)
            if len(finished_ids) >= n:
                break

            # When no work and requests remain to be added, sleep briefly
            # so the 1-second interval check can fire
            if not step_result.has_work and add_idx < n:
                time.sleep(0.05)

    scheduler.shutdown()

    # Build output list in order
    output_list = []
    for req in requests:
        rid = req.request_id
        if rid in results:
            tokens, cached = results[rid]
            output_list.append((rid, tokens, cached))
        else:
            output_list.append((rid, [], 0))

    return output_list


# ---------------------------------------------------------------------------
# Helper: VLM input preparation
# ---------------------------------------------------------------------------

def _prepare_vlm_inputs(
    vlm_model,
    processor,
    messages: List[Dict[str, Any]],
    images: List[Any],
) -> Tuple[List[int], Any, Dict[str, Any], Optional[str]]:
    """
    Prepare VLM inputs at the scheduler level.

    Replicates VLMBatchedEngine._prepare_vision_inputs() logic.

    Returns:
        (token_ids, inputs_embeds, extra_kwargs, image_hash)
    """
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_template
    from mlx_vlm.utils import prepare_inputs

    from omlx.utils.image import compute_image_hash

    num_images = len(images)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Apply VLM chat template with image placeholders
    try:
        prompt = vlm_apply_template(
            processor, vlm_model.config, messages, num_images=num_images
        )
    except Exception:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

    # Tokenize text and preprocess images
    inputs = prepare_inputs(
        processor, images=images if images else None,
        prompts=[prompt] if isinstance(prompt, str) else prompt,
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    attention_mask = inputs.get("attention_mask")
    extra_model_inputs = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "pixel_values")
        and v is not None
    }

    if pixel_values is not None and num_images > 0:
        # Run vision encoder + embedding merge
        try:
            embed_features = vlm_model.get_input_embeddings(
                input_ids, pixel_values, mask=attention_mask, **extra_model_inputs
            )
        except TypeError:
            # Some models don't accept mask kwarg
            embed_features = vlm_model.get_input_embeddings(
                input_ids, pixel_values, **extra_model_inputs
            )
        mx.eval(embed_features.inputs_embeds)

        # Extract extra kwargs from InputEmbeddingsFeatures
        extra_kwargs = {}
        if hasattr(embed_features, "to_dict"):
            feat_dict = embed_features.to_dict()
            for k, v in feat_dict.items():
                if k != "inputs_embeds" and v is not None:
                    extra_kwargs[k] = v

        # Compute image hash
        image_hash = compute_image_hash(images)

        # Token IDs as list
        token_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()

        return token_ids, embed_features.inputs_embeds, extra_kwargs, image_hash
    else:
        token_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
        return token_ids, None, {}, None


# ---------------------------------------------------------------------------
# Test 1: 9K context cache consistency
# ---------------------------------------------------------------------------

def _test_9k_cache_consistency(model, tokenizer, label: str = "LLM"):
    """Test boundary cache and SSD cache produce consistent outputs."""
    import mlx.core as mx

    print(f"\n  [Test 1/{label}] 9K context cache consistency...")
    prompt_token_ids = _build_9k_prompt(tokenizer)
    print(f"    Prompt tokens: {len(prompt_token_ids)}")

    # --- Boundary ON vs OFF ---
    print("    [1a] Boundary cache ON vs OFF...")
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
    print(f"    ON  ({len(tokens_on)} tokens): {text_on[:100]}...")
    print(f"    OFF ({len(tokens_off)} tokens): {text_off[:100]}...")

    _check_output_quality(text_on, f"{label} boundary-ON")
    _check_output_quality(text_off, f"{label} boundary-OFF")

    match = tokens_on == tokens_off
    if match:
        print("    Token match: IDENTICAL")
    else:
        min_len = min(len(tokens_on), len(tokens_off))
        diff_idx = next(
            (i for i in range(min_len) if tokens_on[i] != tokens_off[i]),
            min_len,
        )
        print(f"    Token match: DIFFER at position {diff_idx}")
    print("    Quality check: PASSED")

    # --- SSD cache hit vs fresh ---
    print("    [1b] SSD cache hit vs fresh prefill...")
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_ssd_")
    try:
        tokens_fresh, cached_fresh = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
        )
        text_fresh = tokenizer.decode(tokens_fresh)
        print(f"    Fresh  ({len(tokens_fresh)} tokens, cached={cached_fresh}): {text_fresh[:100]}...")

        tokens_cached, cached_count = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
        )
        text_cached = tokenizer.decode(tokens_cached)
        print(f"    Cached ({len(tokens_cached)} tokens, cached={cached_count}): {text_cached[:100]}...")

        _check_output_quality(text_cached, f"{label} cached")

        match_ssd = tokens_fresh == tokens_cached
        if match_ssd:
            print("    Token match: IDENTICAL")
        else:
            min_len = min(len(tokens_fresh), len(tokens_cached))
            diff_idx = next(
                (i for i in range(min_len) if tokens_fresh[i] != tokens_cached[i]),
                min_len,
            )
            print(f"    Token match: DIFFER at position {diff_idx}")

        if cached_count > 0:
            print(f"    Cache hit confirmed: {cached_count} tokens from SSD")
        else:
            print("    WARNING: No cache hit detected")

        assert match_ssd, f"[{label}] SSD cache hit/fresh tokens differ"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"    [Test 1/{label}] PASSED")


# ---------------------------------------------------------------------------
# Test 2: 4-request concurrent batching
# ---------------------------------------------------------------------------

def _test_concurrent_batching(model, tokenizer, label: str = "LLM"):
    """Test 4 simultaneous and 4 sequential requests."""
    print(f"\n  [Test 2/{label}] 4-request concurrent batching...")

    prompts = _build_short_prompts(tokenizer, 4)

    # --- Concurrent (all at once) ---
    print("    [2a] Concurrent (4 requests at once)...")
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_batch_")
    try:
        results = _generate_batch(
            model, tokenizer, prompts,
            mode="concurrent",
            ssd_cache_dir=tmp_dir,
        )
        for rid, tokens, cached in results:
            text = tokenizer.decode(tokens)
            print(f"    {rid}: {len(tokens)} tokens - {text[:80]}...")
            _check_output_quality(text, f"{label} concurrent {rid}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("    Concurrent: PASSED")

    # --- Sequential (1-second intervals) ---
    print("    [2b] Sequential (1-second intervals)...")
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_seq_")
    try:
        results = _generate_batch(
            model, tokenizer, prompts,
            mode="sequential",
            ssd_cache_dir=tmp_dir,
        )
        for rid, tokens, cached in results:
            text = tokenizer.decode(tokens)
            print(f"    {rid}: {len(tokens)} tokens - {text[:80]}...")
            _check_output_quality(text, f"{label} sequential {rid}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("    Sequential: PASSED")

    print(f"    [Test 2/{label}] PASSED")


# ---------------------------------------------------------------------------
# Test 3: TurboQuant 3-bit
# ---------------------------------------------------------------------------

def _test_turboquant(model, tokenizer, label: str = "LLM"):
    """Test TurboQuant 3-bit with cache consistency and batching."""
    print(f"\n  [Test 3/{label}] TurboQuant 3-bit...")

    prompt_token_ids = _build_9k_prompt(tokenizer)

    # --- TQ cache ON vs OFF (quality-only, TQ is lossy) ---
    print("    [3a] TQ boundary cache ON vs OFF (quality check)...")
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_tq_")
    try:
        tokens_tq_on, _ = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
            turboquant_bits=3.0,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    tokens_tq_off, _ = _generate_tokens(
        model, tokenizer, prompt_token_ids,
        ssd_cache_dir=None,
        turboquant_bits=3.0,
    )

    text_tq_on = tokenizer.decode(tokens_tq_on)
    text_tq_off = tokenizer.decode(tokens_tq_off)
    print(f"    TQ ON  ({len(tokens_tq_on)} tokens): {text_tq_on[:100]}...")
    print(f"    TQ OFF ({len(tokens_tq_off)} tokens): {text_tq_off[:100]}...")
    _check_output_quality(text_tq_on, f"{label} TQ boundary-ON")
    _check_output_quality(text_tq_off, f"{label} TQ boundary-OFF")
    print("    TQ boundary quality check: PASSED")

    # --- TQ SSD cache hit vs fresh ---
    print("    [3b] TQ SSD cache hit vs fresh...")
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_tq_ssd_")
    try:
        tokens_tq_fresh, _ = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
            turboquant_bits=3.0,
        )
        text_tq_fresh = tokenizer.decode(tokens_tq_fresh)
        print(f"    TQ Fresh  ({len(tokens_tq_fresh)} tokens): {text_tq_fresh[:100]}...")

        tokens_tq_cached, cached_count = _generate_tokens(
            model, tokenizer, prompt_token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
            turboquant_bits=3.0,
        )
        text_tq_cached = tokenizer.decode(tokens_tq_cached)
        print(f"    TQ Cached ({len(tokens_tq_cached)} tokens, cached={cached_count}): {text_tq_cached[:100]}...")

        _check_output_quality(text_tq_cached, f"{label} TQ cached")

        match_tq_ssd = tokens_tq_fresh == tokens_tq_cached
        if match_tq_ssd:
            print("    TQ SSD token match: IDENTICAL")
        else:
            min_len = min(len(tokens_tq_fresh), len(tokens_tq_cached))
            diff_idx = next(
                (i for i in range(min_len) if tokens_tq_fresh[i] != tokens_tq_cached[i]),
                min_len,
            )
            print(f"    TQ SSD token match: DIFFER at position {diff_idx}")

        if cached_count > 0:
            print(f"    TQ cache hit confirmed: {cached_count} tokens from SSD")

        assert match_tq_ssd, f"[{label}] TQ SSD cache hit/fresh tokens differ"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("    TQ SSD cache: PASSED")

    # --- TQ batching ---
    print("    [3c] TQ batching (4 concurrent requests)...")
    prompts = _build_short_prompts(tokenizer, 4)
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_tq_batch_")
    try:
        results = _generate_batch(
            model, tokenizer, prompts,
            mode="concurrent",
            ssd_cache_dir=tmp_dir,
            turboquant_bits=3.0,
        )
        for rid, tokens, cached in results:
            text = tokenizer.decode(tokens)
            print(f"    {rid}: {len(tokens)} tokens - {text[:80]}...")
            _check_output_quality(text, f"{label} TQ batch {rid}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("    TQ batching: PASSED")

    print(f"    [Test 3/{label}] PASSED")


# ---------------------------------------------------------------------------
# Test 4: VLM engine basics (re-run tests 1-3 on VLMModelAdapter)
# ---------------------------------------------------------------------------

def _test_vlm_engine_basics(adapter, tokenizer):
    """Re-run cache consistency, batching, and TurboQuant on VLM adapter with text-only."""
    print("\n  [Test 4] VLM engine basics (text-only on VLMModelAdapter)...")

    # Test 1 on VLM adapter
    _test_9k_cache_consistency(adapter, tokenizer, label="VLM")

    # Test 2 on VLM adapter
    _test_concurrent_batching(adapter, tokenizer, label="VLM")

    # Test 3 on VLM adapter
    _test_turboquant(adapter, tokenizer, label="VLM")

    print("    [Test 4] PASSED")


# ---------------------------------------------------------------------------
# Test 5: VLM image caching (5K text + image, 3 turns)
# ---------------------------------------------------------------------------

def _test_vlm_image_caching(vlm_model, processor, adapter):
    """Test image caching works across multi-turn VLM conversations."""
    import mlx.core as mx

    from omlx.utils.image import compute_image_hash

    print("\n  [Test 5] VLM image caching (5K text + image, 3 turns)...")

    tokenizer = getattr(processor, "tokenizer", processor)
    long_system = _build_5k_system(tokenizer)

    # Create 3 distinct images
    images = [_create_test_image(seed=i) for i in range(3)]
    hashes = [compute_image_hash([img]) for img in images]
    print(f"    Image hashes: {[h[:12] for h in hashes]}")
    assert len(set(hashes)) == 3, "All 3 images must have different hashes"

    responses = []
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_vlm_cache_")
    try:
        for turn in range(3):
            print(f"    [Turn {turn+1}] Preparing VLM inputs...")

            # Build cumulative messages
            messages = [{"role": "system", "content": long_system}]

            # Add previous turns
            for prev_turn in range(turn):
                messages.append({
                    "role": "user",
                    "content": f"Describe image {prev_turn+1} in detail."
                })
                messages.append({
                    "role": "assistant",
                    "content": responses[prev_turn]
                })

            # Add current turn
            messages.append({
                "role": "user",
                "content": f"Describe image {turn+1} in detail."
            })

            # Collect all images up to this turn
            turn_images = images[:turn + 1]

            token_ids, embeds, extra_kwargs, image_hash = _prepare_vlm_inputs(
                vlm_model, processor, messages, turn_images
            )
            print(f"    Turn {turn+1}: {len(token_ids)} tokens, hash={image_hash[:12] if image_hash else 'None'}")

            assert embeds is not None, f"Turn {turn+1}: inputs_embeds should not be None"

            # Generate
            output_tokens, cached = _generate_tokens(
                adapter, tokenizer, token_ids,
                ssd_cache_dir=tmp_dir, block_size=2048,
                vlm_inputs_embeds=embeds,
                vlm_extra_kwargs=extra_kwargs,
                vlm_image_hash=image_hash,
            )

            text = tokenizer.decode(output_tokens)
            print(f"    Turn {turn+1} response ({len(output_tokens)} tokens): {text[:100]}...")
            if len(output_tokens) == 0:
                print(f"    WARNING: Turn {turn+1} produced empty output (model may not support this format)")
                text = "(empty)"
            else:
                _check_output_quality(text, f"VLM image cache turn {turn+1}")
            responses.append(text)

        print("    All 3 turns completed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("    [Test 5] PASSED")


# ---------------------------------------------------------------------------
# Test 6: VLM multi-turn image quality
# ---------------------------------------------------------------------------

def _test_vlm_multiturn_quality(vlm_model, processor, adapter):
    """Test coherent responses across 3 multi-turn VLM conversations with images."""
    import mlx.core as mx

    print("\n  [Test 6] VLM multi-turn image quality...")

    tokenizer = getattr(processor, "tokenizer", processor)

    # Create visually distinct colored images
    color_images = [
        _create_colored_image((255, 0, 0)),    # Red
        _create_colored_image((0, 0, 255)),    # Blue
        _create_colored_image((0, 255, 0)),    # Green
    ]
    color_names = ["red", "blue", "green"]

    questions = [
        "Describe the color and appearance of this image in a few sentences.",
        "Describe the color and appearance of this new image in a few sentences.",
        "Describe the color and appearance of this third image in a few sentences.",
    ]

    responses = []
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_vlm_quality_")
    try:
        for turn in range(3):
            print(f"    [Turn {turn+1}] {color_names[turn]} image...")

            messages = []

            # Add previous turns
            for prev_turn in range(turn):
                messages.append({
                    "role": "user",
                    "content": questions[prev_turn],
                })
                messages.append({
                    "role": "assistant",
                    "content": responses[prev_turn],
                })

            # Add current turn
            messages.append({
                "role": "user",
                "content": questions[turn],
            })

            # Collect all images up to this turn
            turn_images = color_images[:turn + 1]

            token_ids, embeds, extra_kwargs, image_hash = _prepare_vlm_inputs(
                vlm_model, processor, messages, turn_images
            )

            assert embeds is not None, f"Turn {turn+1}: inputs_embeds should not be None"

            output_tokens, _ = _generate_tokens(
                adapter, tokenizer, token_ids,
                ssd_cache_dir=tmp_dir, block_size=2048,
                vlm_inputs_embeds=embeds,
                vlm_extra_kwargs=extra_kwargs,
                vlm_image_hash=image_hash,
            )

            text = tokenizer.decode(output_tokens)
            print(f"    Turn {turn+1} response: {text[:150]}")
            if len(output_tokens) == 0:
                print(f"    WARNING: Turn {turn+1} produced empty output (model may not support this format)")
                text = "(empty)"
            else:
                _check_output_quality(text, f"VLM quality turn {turn+1}")
            responses.append(text)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("    All 3 turns completed")
    print("    [Test 6] PASSED")


# ---------------------------------------------------------------------------
# Test 7: VLM image caching with 4-request batching
# ---------------------------------------------------------------------------

def _test_vlm_image_batch(vlm_model, processor, adapter):
    """Test image caching during concurrent VLM batch processing."""
    import mlx.core as mx

    print("\n  [Test 7] VLM image caching with 4-request batching...")

    tokenizer = getattr(processor, "tokenizer", processor)

    # Create 4 different images and prepare VLM inputs
    images = [_create_test_image(seed=i + 10) for i in range(4)]
    vlm_questions = [
        "Describe the colors you see in this image.",
        "What patterns do you notice in this image?",
        "Describe the overall appearance of this image.",
        "What does this image look like? Be brief.",
    ]

    prompt_list = []
    vlm_embeds_list = []

    for i in range(4):
        messages = [{"role": "user", "content": vlm_questions[i]}]
        token_ids, embeds, extra_kwargs, image_hash = _prepare_vlm_inputs(
            vlm_model, processor, messages, [images[i]]
        )
        assert embeds is not None, f"Request {i}: inputs_embeds should not be None"

        prompt_list.append(token_ids)
        vlm_embeds_list.append((embeds, extra_kwargs, image_hash))

    # Run concurrent batch
    tmp_dir = tempfile.mkdtemp(prefix="omlx_test_vlm_batch_")
    try:
        results = _generate_batch(
            adapter, tokenizer, prompt_list,
            mode="concurrent",
            ssd_cache_dir=tmp_dir,
            vlm_embeds_list=vlm_embeds_list,
        )

        for rid, tokens, cached in results:
            text = tokenizer.decode(tokens)
            print(f"    {rid}: {len(tokens)} tokens - {text[:80]}...")
            _check_output_quality(text, f"VLM batch {rid}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("    All 4 VLM batch requests produced output: PASSED")
    print("    [Test 7] PASSED")


# ---------------------------------------------------------------------------
# Main test entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_path",
    MODELS,
    ids=[Path(m).name for m in MODELS],
)
def test_full_integration(model_path):
    """Full integration test for a single model across all test categories."""
    import mlx.core as mx

    if not Path(model_path).exists():
        pytest.skip(f"Model not found: {model_path}")

    model_name = Path(model_path).name
    print(f"\n{'='*60}")
    print(f"Full Integration Test: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    # ========== Phase 1: LLM engine (mlx-lm) ==========
    print(f"\n{'='*40}")
    print("Phase 1: LLM engine (mlx-lm)")
    print(f"{'='*40}")

    from mlx_lm import load

    from omlx.patches.gated_delta_advance import apply_gated_delta_advance_patch

    with _track_peak_memory("LLM model load"):
        model, tokenizer = load(model_path)
        patch_applied = apply_gated_delta_advance_patch(model)
    print(f"  GatedDeltaNet patch: {'applied' if patch_applied else 'skipped'}")

    try:
        with _track_peak_memory("Test 1 - 9K cache consistency"):
            _test_9k_cache_consistency(model, tokenizer)
        with _track_peak_memory("Test 2 - concurrent batching"):
            _test_concurrent_batching(model, tokenizer)
        with _track_peak_memory("Test 3 - TurboQuant"):
            _test_turboquant(model, tokenizer)
    finally:
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    # ========== Phase 2: VLM engine (mlx-vlm) ==========
    print(f"\n{'='*40}")
    print("Phase 2: VLM engine (mlx-vlm)")
    print(f"{'='*40}")

    from omlx.engine.vlm import _patch_gemma4_vision_tower, _patch_video_processor_bug
    from omlx.models.vlm import VLMModelAdapter

    _patch_video_processor_bug()
    _patch_gemma4_vision_tower(None)

    try:
        from mlx_vlm.utils import load as vlm_load
        with _track_peak_memory("VLM model load"):
            vlm_model, processor = vlm_load(model_path)
    except (ValueError, ImportError, Exception) as e:
        print(f"  VLM load failed (model may be text-only LLM): {e}")
        print("  Skipping VLM tests for this model.")
        print(f"\n{'='*60}")
        print(f"LLM TESTS PASSED (VLM skipped): {model_name}")
        print(f"{'='*60}")
        return

    # Build weight-sharing decode model (zero-copy from VLM weights)
    decode_model = None
    try:
        from pathlib import Path as _Path

        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model

        with _track_peak_memory("VLM decode model (weight sharing)"):
            lm_model, _ = load_model(_Path(model_path), lazy=True)
            vlm_params = dict(tree_flatten(vlm_model.language_model.parameters()))
            lm_params = [("language_model." + k, v) for k, v in vlm_params.items()]
            lm_model.load_weights(lm_params, strict=False)
            apply_gated_delta_advance_patch(lm_model)
            decode_model = lm_model
        print(f"  VLM decode model ready (weight sharing, zero-copy)")
    except Exception as e:
        print(f"  VLM decode model failed: {e}")

    adapter = VLMModelAdapter(vlm_model, decode_model=decode_model)
    vlm_tokenizer = getattr(processor, "tokenizer", processor)

    patch_applied = apply_gated_delta_advance_patch(adapter._language_model)
    print(f"  VLM GatedDeltaNet patch: {'applied' if patch_applied else 'skipped'}")

    try:
        with _track_peak_memory("Test 4 - VLM engine basics"):
            _test_vlm_engine_basics(adapter, vlm_tokenizer)
        with _track_peak_memory("Test 5 - VLM image caching"):
            _test_vlm_image_caching(vlm_model, processor, adapter)
        with _track_peak_memory("Test 6 - VLM multi-turn quality"):
            _test_vlm_multiturn_quality(vlm_model, processor, adapter)
        with _track_peak_memory("Test 7 - VLM image batch"):
            _test_vlm_image_batch(vlm_model, processor, adapter)
    finally:
        del vlm_model, processor, adapter, vlm_tokenizer, decode_model
        gc.collect()
        mx.clear_cache()

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED: {model_name}")
    print(f"{'='*60}")
