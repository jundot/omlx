#!/usr/bin/env python3
"""Read-only official-weight execution smoke for DFly2 fusion/correction heads.

Run on the parent-owned isolated staging directory after shard 4 is present.
No model directory is modified and no generation server is started.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[4]
from omlx.patches.deepseek_v41_dfly2.dfly2_heads import final_norm, hidden_correction, mtp_fusion  # noqa: E402
from omlx.patches.deepseek_v41_dfly2.official_loader import OfficialDFly2Checkpoint  # noqa: E402

HEAD_NAMES = (
    "model.mtp_fusion.eh_proj.weight",
    "model.mtp_fusion.enorm.weight",
    "model.mtp_fusion.gate",
    "model.mtp_fusion.hnorm.weight",
    "model.hidden_correction.down_proj.weight",
    "model.hidden_correction.embed_norm.weight",
    "model.hidden_correction.gate_proj.weight",
    "model.hidden_correction.hidden_norm.weight",
    "model.hidden_correction.up_proj.weight",
    "model.final_norm.weight",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "tests" / "checkpoint_metadata" / "config.json")
    parser.add_argument("--index", type=Path, default=ROOT / "tests" / "checkpoint_metadata" / "model.safetensors.index.json")
    args = parser.parse_args()

    checkpoint = OfficialDFly2Checkpoint.from_metadata(args.config, args.index)
    plan = checkpoint.plan_for_tensor_names(HEAD_NAMES)
    loaded = checkpoint.load_available_shards(args.weights_dir, plan)
    weights = {name: mx.array(value) for name, value in loaded.tensors.items()}
    d = checkpoint.config["text_config"]["hidden_size"]
    eps = checkpoint.config["text_config"]["rms_norm_eps"]
    # Fixed small host inputs make this a decoding-path check, not a benchmark.
    anchor = mx.linspace(-1.0, 1.0, d, dtype=mx.float32)[None, None, :]
    target = mx.linspace(1.0, -1.0, d, dtype=mx.float32)[None, None, :]
    previous = mx.zeros((1, d), dtype=mx.float32)
    fused = mtp_fusion(anchor, target, weights, eps=eps)
    corrected = hidden_correction(fused[:, 0], previous, weights, eps=eps)
    output = final_norm(corrected, weights, eps=eps)
    mx.eval(output)
    finite = mx.all(mx.isfinite(output)).item()
    result = {
        "status": "executed" if finite else "nonfinite-output",
        "shards": list(plan.shards),
        "tensor_count": len(plan.tensor_names),
        "output_shape": list(output.shape),
        "finite": bool(finite),
        "sum": float(mx.sum(output).item()),
        "l2": float(mx.sqrt(mx.sum(output * output)).item()),
    }
    print(json.dumps(result, sort_keys=True))
    if not finite:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
