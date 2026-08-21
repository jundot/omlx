#!/usr/bin/env python3
"""Finalize a part-file conversion (convert_escha_mlx.py --layer/--shared-only)
into a standard MLX checkpoint: shard naming, index.json, config.json
(quantization block), tokenizer + chat template.

Usage: finalize_escha_mlx.py --parts <dir with part-*.safetensors> --out <model dir> [--bits 2] [--src-config <escha config.json>]
"""
import argparse, json, os, struct, shutil, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--gate-bits", type=int, default=8)
    ap.add_argument("--src-config", required=True, help="source Escha config.json (for text_config)")
    ap.add_argument("--tokenizer-dir", required=True, help="dir with working tokenizer files")
    ap.add_argument("--template", default=None, help="chat_template.jinja to copy (Escha's)")
    ap.add_argument("--no-quant", action="store_true", help="skip writing the quantization block")
    args = ap.parse_args()

    part_files = sorted(
        (f for f in os.listdir(args.parts) if f.endswith(".safetensors") and f.startswith("part-")),
        key=lambda f: (f.split("L")[-1].split(".")[0] if f.startswith("part-L") else "zzz"),
    )
    if not part_files:
        sys.exit("no part-*.safetensors files found")
    n = len(part_files)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    weight_map = {}
    total = 0
    for i, pf in enumerate(part_files, 1):
        final = f"model-{i:03d}-of-{n:03d}.safetensors"
        shutil.move(os.path.join(args.parts, pf), os.path.join(args.out, final))
        with open(os.path.join(args.out, final), "rb") as f:
            ln = struct.unpack("<Q", f.read(8))[0]
            total += ln + 8
            hdr = json.loads(f.read(ln))
        for k in hdr:
            if k != "__metadata__":
                weight_map[k] = final
        print("moved", pf, "->", final)

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f)

    cfg = json.load(open(args.src_config))
    cfg.pop("quantization_config", None)
    if args.no_quant:
        cfg.pop("quantization", None)
        json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
        cfg = json.load(open(os.path.join(args.out, "config.json")))
    q = {"group_size": args.group_size, "bits": args.bits, "mode": "affine"}
    n_layers = cfg["text_config"]["num_hidden_layers"]
    for l in range(n_layers):
        q[f"language_model.model.layers.{l}.mlp.switch_mlp.gate_proj"] = {"group_size": 64, "bits": 2}
        q[f"language_model.model.layers.{l}.mlp.switch_mlp.up_proj"] = {"group_size": 64, "bits": 2}
        q[f"language_model.model.layers.{l}.mlp.switch_mlp.down_proj"] = {"group_size": 64, "bits": 3}
        for m in ("mlp.gate", "mlp.shared_expert_gate"):
            q[f"language_model.model.layers.{l}.{m}"] = {"group_size": 64, "bits": args.gate_bits}
    # drop escha-specific knobs mlx does not use
    for k in ("qwen3_5_moe", ):
        pass
    cfg["quantization"] = q
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json"):
        p = os.path.join(args.tokenizer_dir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, f))
    if args.template and os.path.exists(args.template):
        shutil.copy(args.template, os.path.join(args.out, "chat_template.jinja"))
    print(f"finalized {n} shards, {len(weight_map)} tensors -> {args.out}")

if __name__ == "__main__":
    main()
