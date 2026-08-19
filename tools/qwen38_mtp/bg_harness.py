#!/usr/bin/env python
"""Qwen3.8-27B MTP token-exactness harness — drives the SAME BatchGenerator
path the omlx server uses (patched GenerationBatch.next -> MTP draft/verify).

One model load per invocation (never two in one process) so a single 64 GB
machine can hold the 15 GB model comfortably; trajectories are saved to
Q38_OUT and compared with `compare`.

  python tools/qwen38_mtp/harness.py serial N          # serial reference
  python tools/qwen38_mtp/harness.py mtp N [depth]     # MTP decode
  python tools/qwen38_mtp/harness.py compare D1 [D2]   # mtp vs serial jsons

Exit code 1 on any token divergence (compare).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from omlx.patches.mlx_lm_mtp import (  # noqa: E402
    apply_mlx_lm_mtp_patch,
    set_mtp_active,
    set_mtp_depth,
)

apply_mlx_lm_mtp_patch()

MODEL_DIR = os.environ.get("Q38_MERGED", os.path.expanduser("~/qwen38-mtp/merged"))
PROMPT_FILE = os.environ.get(
    "Q38_PROMPT",
    os.path.expanduser("~/qwen38-mtp/challenge/correctness_prompts/"
                       "public_longcopy_gate_english_512.txt"),
)
OUT = os.environ.get("Q38_OUT", os.path.expanduser("~/qwen38-mtp/trajs"))


def load_model(mtp: bool, depth: int):
    set_mtp_active(mtp)
    if depth is not None:
        set_mtp_depth(depth)
    from mlx_lm import load

    t0 = time.time()
    model, tokenizer = load(MODEL_DIR)
    inner = getattr(model, "language_model", model)
    print(
        f"[load] mode={'mtp' if mtp else 'serial'} {time.time()-t0:.1f}s "
        f"mtp_decode_enabled={getattr(inner, '_omlx_mtp_decode_enabled', False)}",
        flush=True,
    )
    return model, tokenizer


def decode(model, tokenizer, max_tokens: int):
    from mlx_lm.generate import batch_generate

    with open(PROMPT_FILE, encoding="utf-8") as f:
        prompt = f.read().strip()
    ids = tokenizer.encode(prompt)
    t0 = time.time()
    resp = batch_generate(
        model,
        tokenizer,
        prompts=[ids],
        max_tokens=[max_tokens],
        return_token_ids=True,
        verbose=False,
    )
    dt = time.time() - t0
    toks = resp.token_ids[0]
    print(
        f"[decode] {max_tokens} tok in {dt:.1f}s ({dt / max_tokens * 1000:.1f} ms/tok)",
        flush=True,
    )
    return toks, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serial", "mtp", "compare"])
    ap.add_argument("arg", nargs="?", default="96")
    ap.add_argument("arg2", nargs="?", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.cmd == "compare":
        serial = json.load(open(f"{OUT}/serial.json"))["ids"]
        files = [args.arg] + ([args.arg2] if args.arg2 else [])
        bad = False
        for f in files:
            cand = json.load(open(f"{OUT}/{f}"))["ids"]
            n = min(len(serial), len(cand))
            mism = [i for i in range(n) if serial[i] != cand[i]]
            ok = not mism and len(serial) == len(cand)
            print(
                f"{f}: serial={len(serial)} cand={len(cand)} mism={len(mism)} "
                f"first={mism[:5]} {'TOKEN-EXACT' if ok else 'DIVERGED'}"
            )
            bad = bad or not ok
        sys.exit(1 if bad else 0)

    ntok = int(args.arg)
    if args.cmd == "serial":
        model, tok = load_model(False, None)
        toks, dt = decode(model, tok, ntok)
        json.dump({"ids": toks, "ms_tok": dt / ntok * 1000},
                  open(f"{OUT}/serial.json", "w"))
        return

    depth = int(args.arg2) if args.arg2 else 2
    model, tok = load_model(True, depth)
    toks, dt = decode(model, tok, ntok)
    fname = f"mtp_d{depth}.json"
    json.dump({"ids": toks, "ms_tok": dt / ntok * 1000, "depth": depth},
              open(f"{OUT}/{fname}", "w"))
    print(f"mtp_d{depth} ids[:12]={toks[:12]} len={len(toks)}")


if __name__ == "__main__":
    main()