#!/usr/bin/env python3
"""Read-only verifier for an official DFly2 weight staging directory.

Run this beside the staged download, outside any model directory. It neither downloads
nor writes to model directories. The default inspection uses safetensors slices,
so it validates names, dtypes and shapes without materialising multi-GB tensors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
from omlx.patches.deepseek_v41_dfly2.official_loader import (  # noqa: E402
    OfficialDFly2Checkpoint,
    draft_layer_ids,
    required_prefixes,
    target_layer_map,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "tests" / "checkpoint_metadata" / "config.json")
    parser.add_argument("--index", type=Path, default=ROOT / "tests" / "checkpoint_metadata" / "model.safetensors.index.json")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                    help="checkpoint layer indices; default: every draft layer in dfly_config")
    parser.add_argument("--load-synthetic-compatible", action="store_true", help="materialise selected tensors; do not use for the full layer prefix")
    args = parser.parse_args()

    checkpoint = OfficialDFly2Checkpoint.from_metadata(args.config, args.index)
    layers = tuple(args.layers) if args.layers else draft_layer_ids(checkpoint.config)
    mapping = target_layer_map(checkpoint.config)
    print('draft->target', {index: mapping[index] for index in layers})
    plan = checkpoint.plan_for_prefixes(required_prefixes(layer_ids=layers, include_fusion=True))
    from safetensors import safe_open

    records = []
    for name in plan.tensor_names:
        path = args.weights_dir / checkpoint.weight_map[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework="numpy") as handle:
            if name not in handle.keys():
                raise ValueError(f"index claims {name} is in {path.name}, but it is absent")
            view = handle.get_slice(name)
            records.append({"name": name, "shard": path.name, "dtype": view.get_dtype(), "shape": list(view.get_shape())})

    result = {
        "status": "header-validated",
        "dfly": checkpoint.config["dfly_config"],
        "layers": args.layers,
        "tensor_count": len(records),
        "shards": list(plan.shards),
        "tensors": records,
    }
    if args.load_synthetic_compatible:
        loaded = checkpoint.load_available_shards(args.weights_dir, plan)
        result["materialised_tensor_count"] = len(loaded.tensors)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
