#!/usr/bin/env python
"""Build the merged Qwen3.8-27B checkpoint that omlx's Lightning MTP path
loads: the 4-bit backbone + the bf16 MTP head with its 15 tensors prefixed
``mtp.`` (the Swift loader's merge contract). Backbone shards are symlinked;
the head tensors are written as an extra ``model-00004-of-00004.safetensors``
shard so mlx-lm's ``model*.safetensors`` glob picks them up.

Expects the digest-verified trees under ``~/qwen38-mtp/{backbone,head}``
(download.sh + fixtures digest check) and writes ``~/qwen38-mtp/merged``.

Usage:
    python tools/qwen38_mtp/merge_checkpoint.py
"""
from __future__ import annotations

import json
import os
import shutil

import mlx.core as mx

BACKBONE = os.environ.get("Q38_BACKBONE", os.path.expanduser("~/qwen38-mtp/backbone"))
HEAD = os.environ.get("Q38_HEAD", os.path.expanduser("~/qwen38-mtp/head"))
OUT = os.environ.get("Q38_MERGED", os.path.expanduser("~/qwen38-mtp/merged"))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name in sorted(os.listdir(BACKBONE)):
        if name == ".cache":
            continue
        dst = os.path.join(OUT, name)
        if os.path.exists(dst):
            continue
        if name.endswith(".safetensors") and name != "model.safetensors.index.json":
            os.symlink(os.path.join(BACKBONE, name), dst)
        else:
            shutil.copyfile(os.path.join(BACKBONE, name), dst)

    with open(os.path.join(BACKBONE, "model.safetensors.index.json")) as f:
        idx = json.load(f)

    outs = mx.load(os.path.join(HEAD, "model.safetensors"), format="safetensors")
    prefixed = {"mtp." + k: v for k, v in outs.items()}
    total = sum(v.nbytes for v in prefixed.values())
    shard = "model-00004-of-00004.safetensors"
    mx.save_safetensors(
        os.path.join(OUT, shard),
        prefixed,
        metadata={"total_size": str(total)},
    )
    for k in prefixed:
        idx["weight_map"][k] = shard
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=1)

    n_mtp = sum(1 for k in idx["weight_map"] if k.startswith("mtp."))
    print(f"merged: {len(idx['weight_map'])} keys, {n_mtp} mtp.* keys -> {OUT}")
    assert n_mtp == 15, f"expected 15 mtp.* tensors, got {n_mtp}"


if __name__ == "__main__":
    main()
