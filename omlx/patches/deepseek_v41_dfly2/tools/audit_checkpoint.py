#!/usr/bin/env python3
"""Verify the DFly2 checkpoint contract against a target-model config/index.

No weights are loaded. This intentionally checks the serving seam before any
large local download or production-side conversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_DFLY = {
    "block_size",
    "target_kv_layers",
    "target_index_layers",
    "index_topk",

    "mtp_fusion",
    "hidden_correction",
    "attentive_correction",
}
SHAPE_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def text_config(config: dict) -> dict:
    return config.get("text_config") or config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--draft-index", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--target-index", type=Path, required=True)
    args = parser.parse_args()

    draft = read_json(args.draft_config)
    draft_index = read_json(args.draft_index)
    target = read_json(args.target_config)
    target_index = read_json(args.target_index)
    dfly = draft.get("dfly_config")
    if not isinstance(dfly, dict):
        raise SystemExit("draft config has no dfly_config")
    missing = sorted(REQUIRED_DFLY - set(dfly))
    if missing:
        raise SystemExit(f"draft config missing DFly2 fields: {missing}")

    draft_text, target_text = text_config(draft), text_config(target)
    # The AngelSpec training config names this `context_local_window`; the
    # released HF checkpoint omits it and inherits the target's
    # `text_config.sliding_window` (128 for this DeepSeek checkpoint).
    local_window = dfly.get("context_local_window", draft_text.get("sliding_window"))
    if local_window is None:
        raise SystemExit("cannot derive DFly2 local window from draft config")
    mismatches = {
        name: (draft_text.get(name), target_text.get(name))
        for name in SHAPE_KEYS
        if draft_text.get(name) != target_text.get(name)
    }
    draft_weights = draft_index.get("weight_map", {})
    target_weights = target_index.get("weight_map", {})
    target_dfly_weights = [name for name in target_weights if "dfly" in name.lower()]
    target_mtp_weights = [name for name in target_weights if "mtp" in name.lower()]

    report = {
        "compatible_target_shape": not mismatches,
        "shape_mismatches": mismatches,
        "official_dfly": {
            "block_size": dfly["block_size"],
            "draft_tokens_per_step": dfly["block_size"] - 1,
            "target_kv_layers": dfly["target_kv_layers"],
            "target_index_layers": dfly["target_index_layers"],
            "index_topk": dfly["index_topk"],
            "context_local_window": local_window,
            "local_window_source": (
                "dfly_config.context_local_window"
                if "context_local_window" in dfly
                else "text_config.sliding_window"
            ),
            "candidate_slots_before_dedupe": dfly["index_topk"]
            + local_window
            + dfly["block_size"],
            "weight_count": len(draft_weights),
            "total_bytes": draft_index.get("metadata", {}).get("total_size"),
        },
        "target": {
            "model_type": target.get("model_type"),
            "architecture": target.get("architectures"),
            "weight_count": len(target_weights),
            "dfly_config_present": bool(target.get("dfly_config"))
            or any(key.lower().startswith("dfly_") for key in target),
            "dfly_weight_count": len(target_dfly_weights),
            "mtp_weight_count": len(target_mtp_weights),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
