# SPDX-License-Identifier: Apache-2.0
"""Live behavioral acceptance test for MoE expert offloading.

Loads a real MoE checkpoint twice — offloaded at 25% residency through the
same lazy-load -> wrap -> materialize sequence the batched engine runs, then
fully resident — and requires:

* the offloaded Metal footprint stays a fraction of the resident one, and
  even its PEAK stays well under the resident size (peak is the proof that
  non-resident experts were never materialized), and
* greedy generations match the resident model. Cache capacity changes
  gather/reduction order, so a paraphrase-level fork late in a generation is
  legitimate (deterministic at fixed capacity, never early); wrong-expert
  bugs show as garbage from the first tokens instead. Short completions on
  this hardware/mlx version come out bit-identical; the assertion allows one
  fork across the four prompts so a future kernel-tiling change does not
  read as a feature regression.

Needs ~15 GB free memory and the checkpoint in the local HF cache; skipped
otherwise.
"""

import gc

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

MODEL_REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"


def _model_cached() -> bool:
    if not HAS_MLX:
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_REPO, allow_patterns=["*.safetensors"], local_files_only=True
        )
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _model_cached(), reason=f"{MODEL_REPO} not cached"),
]

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that reverses a linked list.",
    "Name three prime numbers between 80 and 110.",
    "What is the capital of Australia, and which larger city is it often confused with?",
]
MAX_TOKENS = 96
RESIDENT_FRACTION = 0.25


def _generate_all(model, tok):
    from mlx_lm import generate

    outs = []
    for p in PROMPTS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        outs.append(
            generate(model, tok, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
        )
    return outs


def _teardown(model, tok):
    del model, tok
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def test_offloaded_generation_matches_resident():
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        moe_offload_stats,
    )
    from omlx.utils.model_loading import lm_load_compat, materialize_lazy_state

    # Offloaded arm first: its peak sample must precede the resident load,
    # because mx.get_peak_memory() is a process-global high-water mark.
    model, tok = lm_load_compat(MODEL_REPO, lazy=True)
    wrapped = apply_moe_expert_offload(model, MODEL_REPO, RESIDENT_FRACTION)
    assert wrapped > 0, "no layers wrapped — arm would silently run resident"
    materialize_lazy_state(model)
    offload_texts = _generate_all(model, tok)
    offload_active = mx.get_active_memory()
    offload_peak = mx.get_peak_memory()
    stats = moe_offload_stats(model)
    _teardown(model, tok)

    assert stats["layers"] == wrapped
    assert stats["misses"] > 0, "cache never exercised"
    assert 0.4 < stats["hit_rate"] < 0.98, stats

    model, tok = lm_load_compat(MODEL_REPO)
    resident_texts = _generate_all(model, tok)
    resident_active = mx.get_active_memory()
    _teardown(model, tok)

    assert offload_active < 0.55 * resident_active, (offload_active, resident_active)
    assert offload_peak < 0.75 * resident_active, (offload_peak, resident_active)

    for t in offload_texts:
        assert t and t.strip(), "empty offloaded generation"
    identical = sum(a == b for a, b in zip(resident_texts, offload_texts))
    assert identical >= len(PROMPTS) - 1, (
        f"only {identical}/{len(PROMPTS)} generations identical to resident — "
        "beyond rounding-fork territory"
    )
