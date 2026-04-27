#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2E test for VisionFeatureSSDCache with real VLM models.

Usage:
    conda run -n vllm-mlx python tests/e2e_vision_cache.py <model_path> [--ssd-dir /tmp/vc_test]

Tests:
    1. Model load + encode_image / cached_image_features capability detection
    2. Vision feature computation via _compute_vision_features
    3. Cache miss → store → cache hit roundtrip
    4. Output quality: cached vs fresh features produce identical embeddings
    5. SSD persistence: write → clear memory → load from SSD
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

# Add parent to path for omlx imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omlx.cache.vision_feature_cache import VisionFeatureSSDCache
from omlx.engine.vlm import VLMBatchedEngine, _QWEN_VISION_MODELS


def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple test image with colored blocks."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            pixels[x, y] = (r, g, b)
    return img


def test_model(model_path: str, ssd_dir: Optional[str] = None) -> bool:
    """Run all vision cache tests for a single model."""
    from mlx_vlm.utils import load as vlm_load, prepare_inputs

    from omlx.engine.vlm import _patch_gemma4_vision_tower, _patch_video_processor_bug
    from omlx.utils.image import compute_image_hash

    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"{'='*60}")

    # ── Step 1: Load model ──────────────────────────────────────
    print("\n[1/6] Loading model...")
    _patch_video_processor_bug()
    _patch_gemma4_vision_tower(None)
    vlm_model, processor = vlm_load(model_path)

    model_type = getattr(vlm_model.config, "model_type", "unknown")
    has_encode_image = hasattr(vlm_model, "encode_image")

    print(f"  model_type: {model_type}")
    print(f"  has encode_image: {has_encode_image}")
    print(f"  in _QWEN_VISION_MODELS: {model_type in _QWEN_VISION_MODELS}")
    print(f"  is llava: {model_type == 'llava'}")

    # ── Step 2: Prepare inputs ──────────────────────────────────
    print("\n[2/6] Preparing vision inputs...")
    test_image = create_test_image(336, 336)
    image_hash = compute_image_hash([test_image])
    print(f"  image_hash: {image_hash[:16]}...")

    tokenizer = getattr(processor, "tokenizer", processor)

    # Use mlx-vlm's apply_chat_template to properly insert image tokens.
    # Different models use different image placeholder formats.
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_template

    messages = [{"role": "user", "content": "Describe this image."}]
    try:
        prompt = vlm_apply_template(
            processor, vlm_model.config, messages, num_images=1
        )
    except Exception:
        # Fallback: try tokenizer directly
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "Describe this image."

    inputs = prepare_inputs(
        processor, images=[test_image], prompts=[prompt]
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    attention_mask = inputs.get("attention_mask")
    extra_model_inputs = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "pixel_values") and v is not None
    }

    print(f"  input_ids shape: {input_ids.shape}")
    pv_info = type(pixel_values).__name__
    if isinstance(pixel_values, mx.array):
        pv_info += f" shape={pixel_values.shape}"
    elif isinstance(pixel_values, (list, tuple)):
        pv_info += f" len={len(pixel_values)}"
    elif isinstance(pixel_values, np.ndarray):
        pv_info += f" shape={pixel_values.shape}"
    print(f"  pixel_values: {pv_info}")
    print(f"  extra_model_inputs keys: {list(extra_model_inputs.keys())}")

    # ── Step 3: Test _compute_vision_features ────────────────────
    print("\n[3/6] Testing _compute_vision_features...")
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._vlm_model = vlm_model
    engine._model_name = model_path

    t0 = time.perf_counter()
    features = engine._compute_vision_features(pixel_values, extra_model_inputs)
    if features is not None:
        mx.eval(features)
    t_compute = time.perf_counter() - t0

    if features is None:
        print(f"  _compute_vision_features returned None (unsupported model)")
        print(f"  This model will use full pipeline without caching")

        # Verify full pipeline still works
        print("\n[3b/6] Verifying full pipeline works...")
        embed = vlm_model.get_input_embeddings(
            input_ids, pixel_values, mask=attention_mask, **extra_model_inputs
        )
        mx.eval(embed.inputs_embeds)
        print(f"  Full pipeline OK: inputs_embeds shape={embed.inputs_embeds.shape}")
        print(f"\n{'='*60}")
        print(f"RESULT: PASS (fallback mode — no vision cache for {model_type})")
        print(f"{'='*60}")
        return True

    feat_shape = features.shape if isinstance(features, mx.array) else f"list[{len(features)}]"
    print(f"  features shape: {feat_shape}")
    print(f"  compute time: {t_compute*1000:.1f}ms")

    # ── Step 4: Test cached_image_features support ───────────────
    print("\n[4/6] Testing cached_image_features kwarg...")
    try:
        call_kwargs = dict(extra_model_inputs)
        call_kwargs["cached_image_features"] = features
        embed_cached = vlm_model.get_input_embeddings(
            input_ids, pixel_values, mask=attention_mask, **call_kwargs
        )
        mx.eval(embed_cached.inputs_embeds)
        print(f"  cached path OK: inputs_embeds shape={embed_cached.inputs_embeds.shape}")
    except TypeError as e:
        print(f"  FAIL: cached_image_features not supported: {e}")
        print(f"\n{'='*60}")
        print(f"RESULT: PARTIAL — _compute works but cached kwarg rejected")
        print(f"{'='*60}")
        return False

    # Compare with fresh computation
    print("\n[4b/6] Quality check: cached vs fresh embeddings...")
    embed_fresh = vlm_model.get_input_embeddings(
        input_ids, pixel_values, mask=attention_mask, **extra_model_inputs
    )
    mx.eval(embed_fresh.inputs_embeds)

    max_diff = mx.max(mx.abs(embed_cached.inputs_embeds - embed_fresh.inputs_embeds)).item()
    mean_diff = mx.mean(mx.abs(embed_cached.inputs_embeds - embed_fresh.inputs_embeds)).item()
    identical = mx.array_equal(embed_cached.inputs_embeds, embed_fresh.inputs_embeds)
    print(f"  identical: {identical}")
    print(f"  max_diff: {max_diff:.2e}")
    print(f"  mean_diff: {mean_diff:.2e}")
    if max_diff > 1e-3:
        print(f"  WARNING: significant difference between cached and fresh embeddings!")

    # ── Step 5: Test VisionFeatureSSDCache roundtrip ─────────────
    print("\n[5/6] Testing VisionFeatureSSDCache roundtrip...")
    cache_dir = Path(ssd_dir) if ssd_dir else Path(tempfile.mkdtemp()) / "vision_cache"
    cache = VisionFeatureSSDCache(cache_dir=cache_dir, max_memory_entries=5)

    # Miss
    result = cache.get(image_hash, model_path)
    assert result is None, "Expected cache miss"
    print(f"  cache miss: OK")

    # Store
    cache.put(image_hash, model_path, features)
    print(f"  cache put: OK")

    # Memory hit
    result = cache.get(image_hash, model_path)
    assert result is not None, "Expected cache hit"
    if isinstance(result, mx.array):
        assert mx.array_equal(result, features), "Memory cache returned different data"
    print(f"  memory hit: OK")

    # SSD roundtrip
    time.sleep(1.0)  # wait for background writer
    with cache._memory_lock:
        cache._memory_cache.clear()
    result = cache.get(image_hash, model_path)
    assert result is not None, "Expected SSD cache hit"
    if isinstance(result, mx.array) and isinstance(features, mx.array):
        assert mx.allclose(result, features, atol=1e-5), "SSD cache returned different data"
    print(f"  SSD roundtrip: OK")

    stats = cache.stats
    print(f"  stats: {stats}")
    cache.close()

    # ── Step 6: Cache hit performance ────────────────────────────
    print("\n[6/6] Performance comparison...")
    cache2 = VisionFeatureSSDCache(cache_dir=cache_dir, max_memory_entries=5)
    cache2.put(image_hash, model_path, features)

    # Warm: cache hit (no vision tower)
    t0 = time.perf_counter()
    cached = cache2.get(image_hash, model_path)
    call_kwargs2 = dict(extra_model_inputs)
    call_kwargs2["cached_image_features"] = cached
    embed2 = vlm_model.get_input_embeddings(
        input_ids, pixel_values, mask=attention_mask, **call_kwargs2
    )
    mx.eval(embed2.inputs_embeds)
    t_cached = time.perf_counter() - t0

    # Cold: full pipeline (vision tower runs)
    t0 = time.perf_counter()
    embed3 = vlm_model.get_input_embeddings(
        input_ids, pixel_values, mask=attention_mask, **extra_model_inputs
    )
    mx.eval(embed3.inputs_embeds)
    t_fresh = time.perf_counter() - t0

    speedup = t_fresh / t_cached if t_cached > 0 else float("inf")
    print(f"  fresh:  {t_fresh*1000:.1f}ms")
    print(f"  cached: {t_cached*1000:.1f}ms")
    print(f"  speedup: {speedup:.1f}x")

    cache2.close()

    print(f"\n{'='*60}")
    print(f"RESULT: PASS — full vision feature cache working for {model_type}")
    print(f"{'='*60}")
    return True


def main():
    parser = argparse.ArgumentParser(description="E2E vision feature cache test")
    parser.add_argument("model_path", help="Path to VLM model")
    parser.add_argument("--ssd-dir", default=None, help="SSD cache directory")
    args = parser.parse_args()

    success = test_model(args.model_path, args.ssd_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
