# SPDX-License-Identifier: Apache-2.0
"""Paired cold-prefill benchmark for the loaded-instance M4 QMM patch.

Run on a checkout with the native QMM extension built. Keep other GPU traffic
idle; this does not stop/reconfigure a running server or download a checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from omlx.patches.m4_dense_mlp_prefill import apply_m4_dense_mlp_prefill
from omlx.utils.model_loading import (
    apply_post_load_transforms,
    lm_load_compat,
    maybe_apply_pre_load_patches,
)


def _load(path, vlm):
    if not vlm:
        model, tokenizer = lm_load_compat(path, trust_remote_code=True)
        return model, tokenizer
    from mlx_vlm.utils import load

    from omlx.models.vlm import VLMModelAdapter

    maybe_apply_pre_load_patches(path, for_vlm=True)
    vlm_model, processor = load(path)
    # This is the same instance transform used before adapter construction.
    apply_m4_dense_mlp_prefill(vlm_model)
    adapter = VLMModelAdapter(vlm_model)
    return adapter, getattr(processor, "tokenizer", processor)


def _run(model, tokens, step):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    for offset in range(0, len(tokens) - 1, step):
        end = min(offset + step, len(tokens) - 1)
        model(mx.array(tokens[offset:end])[None], cache=cache)
        mx.eval([entry.state for entry in cache])
    output = model(mx.array(tokens[-1:])[None], cache=cache)
    logits = (output.logits if hasattr(output, "logits") else output)[:, -1, :]
    mx.eval(logits)
    result = {
        "seconds": time.perf_counter() - start,
        "tokens": len(tokens),
        "mlx_peak_gib": mx.get_peak_memory() / 1024**3,
    }
    last_logits = mx.array(logits)
    ids = []
    for _ in range(16):
        token = int(mx.argmax(logits, axis=-1).item())
        ids.append(token)
        output = model(mx.array([[token]]), cache=cache)
        logits = (output.logits if hasattr(output, "logits") else output)[:, -1, :]
        mx.eval(logits)
    result["greedy_ids"] = ids
    del cache, output, logits
    gc.collect()
    mx.clear_cache()
    return result, last_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--memory-gib", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.length, args.step, args.repeats, args.memory_gib) <= 0
        or args.cooldown < 0
    ):
        parser.error(
            "positive lengths/repeats/memory and nonnegative cooldown required"
        )
    mx.set_memory_limit(args.memory_gib * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    os.environ.pop("OMLX_M4_DENSE_MLP_PREFILL", None)
    model, tokenizer = _load(str(args.model), args.vlm)
    text = "次の記録を読んで日本語で要約してください。\n" + (
        "監視対象APIは正常です。設定変更は承認後に実施し、障害時には切り戻します。\n"
        * 1000
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    encoded = getattr(encoded, "ids", encoded)
    tokens = list(encoded[: args.length - 32]) + list(encoded[-32:])
    if len(tokens) != args.length:
        raise ValueError("source prompt too short for requested length")
    _run(model, tokens[:1024], args.step)
    os.environ["OMLX_M4_DENSE_MLP_PREFILL"] = "1"
    # Keep the loaded arrays: toggling only the instance classes avoids model
    # reload costs and guarantees both arms read identical weights.
    projections = []
    for _, module in model.named_modules():
        if type(module).__name__ == "MLP":
            for name in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(module, name, None)
                if linear is not None:
                    projections.append((linear, type(linear)))
    apply_post_load_transforms(model, None)
    changed = [
        (linear, original, type(linear))
        for linear, original in projections
        if type(linear) is not original
    ]
    if not changed:
        raise RuntimeError("M4 patch did not route any loaded projections")
    _run(model, tokens[:1024], args.step)
    report = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": mx.device_info()["device_name"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "mlx-vlm")
        },
        "model": str(args.model),
        "config": {
            "length": args.length,
            "step": args.step,
            "repeats": args.repeats,
            "cooldown": args.cooldown,
            "vlm": args.vlm,
        },
        "projections": len(changed),
        "pairs": [],
    }
    try:
        for repeat in range(args.repeats):
            runs, logits = {}, {}
            order = ["stock", "patched"] if repeat % 2 == 0 else ["patched", "stock"]
            for arm in order:
                for linear, original, patched in changed:
                    linear.__class__ = patched if arm == "patched" else original
                runs[arm], logits[arm] = _run(model, tokens, args.step)
                print(json.dumps({"arm": arm, **runs[arm]}), flush=True)
                time.sleep(args.cooldown)
            delta = logits["patched"].astype(mx.float32) - logits["stock"].astype(
                mx.float32
            )
            pair = {
                "repeat": repeat,
                "order": order,
                "runs": runs,
                "logits_nonzero": int(mx.sum(delta != 0).item()),
                "logits_max_abs": float(mx.max(mx.abs(delta)).item()),
                "greedy_match": runs["patched"]["greedy_ids"]
                == runs["stock"]["greedy_ids"],
            }
            report["pairs"].append(pair)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del logits, delta
    finally:
        for linear, original, _ in changed:
            linear.__class__ = original


if __name__ == "__main__":
    main()
