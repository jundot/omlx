#!/usr/bin/env python3
"""Copy alternating shards of a model to a stripe root (multi-SSD expert streaming).

Copies odd-indexed *.safetensors shards from the model dir to a target dir on
a second SSD. Expert reads are then striped across both disks by setting:

    OMLX_EXPERT_STREAMING_EXTRA_ROOTS=/Volumes/SSD 2TB/AI Models/<model-name>

Mirrored shards (files present on the stripe root) are served from there;
the rest fall back to the primary root. The original model dir is never
modified.

Usage:
    .venv/bin/python bench/stripe_model.py \
        --model "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e" \
        --target "/Volumes/SSD 2TB/AI Models/GLM-5.3-Flash-oQ4e" [--verify]
"""

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_mb * 1024**2)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stripe a model's shards across two roots")
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--verify", action="store_true", help="sha256-verify each copied shard")
    args = ap.parse_args()

    src = Path(args.model).expanduser()
    dst = Path(args.target)
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards found in {src}")
    picked = shards[1::2]  # odd-indexed shards go to the stripe root
    print(f"{len(shards)} shards, copying {len(picked)} to {dst}")
    for i, shard in enumerate(shards):
        if i % 2 == 0:
            continue
        target = dst / shard.name
        if target.is_file() and target.stat().st_size == shard.stat().st_size:
            print(f"skip {shard.name} (already present)")
            continue
        print(f"copy {shard.name} ({shard.stat().st_size / 1024**3:.1f} GB)", flush=True)
        tmp = target.with_suffix(".partial")
        shutil.copyfile(shard, tmp)
        if args.verify:
            if sha256(tmp) != sha256(shard):
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"sha256 mismatch for {shard.name}")
        tmp.rename(target)
        print(f"done {shard.name}", flush=True)

    print(
        "\nRun with:\n"
        f"  OMLX_EXPERT_STREAMING_EXTRA_ROOTS={dst} "
        ".venv/bin/python bench/bench_expert_streaming.py --model <key> ..."
    )


if __name__ == "__main__":
    main()