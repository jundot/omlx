"""Explicit local checkpoint smoke test; no checkpoint or private data bundled."""

import argparse
import json
import os
import time

os.environ["OMLX_EXL3_ENABLED"] = "1"
import mlx.core as mx
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template

from omlx.utils.model_loading import maybe_load_custom_quantization

p = argparse.ArgumentParser()
p.add_argument("model")
args = p.parse_args()
mx.set_memory_limit(48 * 1024**3)
mx.set_cache_limit(128 * 1024**2)
t = time.monotonic()
print("LOAD_START", flush=True)
model, processor = maybe_load_custom_quantization(args.model, is_vlm=True)
print(
    json.dumps(
        {
            "loaded_seconds": time.monotonic() - t,
            "active_gib": mx.get_active_memory() / 2**30,
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
    ),
    flush=True,
)
for text in [
    "Reply in one short sentence: what is 2+2?",
    "Write a Python function that returns the unique values in a list while preserving order.",
    "Reply to your mate who claims his rusty hatchback is the fastest car in town. Be funny and brief.",
]:
    prompt = apply_chat_template(
        processor,
        model.config,
        prompt=text,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result = generate(
        model,
        processor,
        prompt=prompt,
        max_tokens=64,
        temperature=0,
        prefill_step_size=64,
        verbose=False,
    )
    print(
        json.dumps(
            {
                "request": text,
                "text": result.text,
                "prompt_tps": result.prompt_tps,
                "decode_tps": result.generation_tps,
                "active_gib": mx.get_active_memory() / 2**30,
            }
        ),
        flush=True,
    )
model.close()
