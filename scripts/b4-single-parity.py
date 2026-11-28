#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B4 — single-Mac parity of the restructured Qwen4-Exp forward (#5033 review).

The review: the pipeline forward now lives in the vendored ``Qwen4ExpModel``
that the single-Mac VLM engine runs, so every single-Mac Qwen4 request goes
through the restructured ``__call__`` — a single-Mac parity run against main
is required before merge.

This instrument runs a TINY synthetic qwen4_exp (the full architecture —
GDN linear attention, QSA full attention, PLE n-gram table, hyper-connections,
MoE — at 2 layers / hidden 32, the test fixture's geometry) with IDENTICAL
weights in two trees and compares greedy generation token-for-token plus the
per-position logits. Weights are transferred through an npz state dict
(``record --save-weights`` in tree A, ``record --load-weights`` in tree B,
strict), so a different parameter-init order between the trees cannot fake a
parity result — and a structural difference fails the strict load loudly.

Usage (the QA-gate two-tree pattern):

    # tree A (the branch under review):
    uv run python scripts/b4-single-parity.py record \
        --out /tmp/b4-a.json --save-weights /tmp/b4-weights.npz
    # tree B (a worktree of upstream/main), same interpreter:
    <venv>/python <worktree>/scripts/b4-single-parity.py record \
        --out /tmp/b4-b.json --load-weights /tmp/b4-weights.npz
    # compare:
    uv run python scripts/b4-single-parity.py compare /tmp/b4-a.json /tmp/b4-b.json

The script must exist in both trees (copy it into the baseline worktree like
the census runner). PASS = identical token ids AND max |Δlogit| <= 1e-4 on
the greedy path (bf16 tiny model, same weights, same kernels — a restructured
forward that computes the same math must match to rounding; the live D18
threshold of 0.2 nats is for cross-rank summation order, not this).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOGIT_TOLERANCE = 1e-4
PROMPT_IDS = [2, 3, 4, 5, 6, 7, 8, 9] * 8  # 64 tokens, fixed
DECODE_STEPS = 32


def _tiny_text_config():
    """The test fixture's geometry (tests/test_mlx_vlm_qwen4_exp_compat.py)."""
    from mlx_vlm.models import qwen4_exp

    return qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )


def _build_model():
    import mlx.core as mx

    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    text = _tiny_text_config()
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
        vocab_size=64,
    )

    mx.random.seed(20261120)
    model = LanguageModel(text, config)
    mx.eval(model.parameters())
    return model


def record(args: argparse.Namespace) -> dict:
    import mlx.core as mx

    model = _build_model()
    if args.ple_mode:
        from mlx_vlm.models.qwen4_exp import language

        language._PLE_RUNTIME_MODE = args.ple_mode
    if args.load_weights:
        weights = mx.load(args.load_weights)
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
    if args.save_weights:
        from mlx.utils import tree_flatten

        flat = dict(tree_flatten(model.parameters()))
        mx.savez(args.save_weights, **flat)

    cache = model.make_cache()
    out = model(mx.array([PROMPT_IDS], dtype=mx.int32), cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    steps = []
    last = logits[0, -1]
    for _ in range(DECODE_STEPS):
        token = int(mx.argmax(last).item())
        steps.append({"token": token, "logits": [float(v) for v in last.tolist()]})
        out = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        mx.eval(logits)
        last = logits[0, -1]
    return {
        "instrument": "b4-single-parity/1",
        "tree_planner": str(Path(sys.modules["omlx"].__file__).resolve()),
        "prompt": PROMPT_IDS,
        "steps": steps,
    }


def compare(a: dict, b: dict) -> dict:
    if a["prompt"] != b["prompt"]:
        return {"verdict": "REFUSED", "reason": "different prompts"}
    if len(a["steps"]) != len(b["steps"]):
        return {"verdict": "REFUSED", "reason": "different step counts"}
    flips = []
    max_delta = 0.0
    for i, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
        if sa["token"] != sb["token"]:
            flips.append({"position": i, "a": sa["token"], "b": sb["token"]})
        for va, vb in zip(sa["logits"], sb["logits"]):
            max_delta = max(max_delta, abs(va - vb))
    verdict = "PASS" if not flips and max_delta <= LOGIT_TOLERANCE else "FAIL"
    return {
        "verdict": verdict,
        "token_flips": flips,
        "max_logit_delta": max_delta,
        "tolerance": LOGIT_TOLERANCE,
        "positions_compared": len(a["steps"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--out", required=True)
    rec.add_argument("--save-weights", default="")
    rec.add_argument("--load-weights", default="")
    rec.add_argument(
        "--ple-mode",
        default="",
        help="force the PLE runtime mode (mmap arms the B3 pool valve)",
    )
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    args = parser.parse_args()

    if args.cmd == "record":
        payload = record(args)
        Path(args.out).write_text(json.dumps(payload) + "\n")
        tokens = [s["token"] for s in payload["steps"]]
        print(f"recorded {len(tokens)} greedy tokens -> {args.out}")
        print(f"tokens: {tokens}")
        return 0
    verdict = compare(
        json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    )
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
