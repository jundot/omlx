#!/usr/bin/env python3
"""Assemble part files from convert_escha_mlx.py into a finished MLX checkpoint:
shard naming, index.json, config.json (quantization block / eschamoe marker),
tokenizer + chat template. Mirrors mlx-community conventions.
"""
import argparse, json, os, shutil, struct, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--gate-bits", type=int, default=8)
    ap.add_argument("--src-config", required=True, help="source Escha config.json")
    ap.add_argument("--tokenizer-dir", required=True, help="dir with working tokenizer files")
    ap.add_argument("--template", default=None, help="chat_template.jinja to copy")
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--expert-format", choices=("affine", "trellis"), default="affine")
    args = ap.parse_args()

    part_files = sorted(
        (f for f in os.listdir(args.parts)
         if f.endswith(".safetensors") and f.startswith("part-")),
        key=lambda f: (f.split("L")[-1].split(".")[0] if f.startswith("part-L") else "zzz"),
    )
    if not part_files:
        sys.exit("no part-*.safetensors files found")
    n = len(part_files)
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
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f)

    cfg = json.load(open(args.src_config))
    cfg.pop("quantization_config", None)
    if args.no_quant:
        cfg.pop("quantization", None)
        json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
        cfg = json.load(open(os.path.join(args.out, "config.json")))
    if args.expert_format == "trellis":
        # Dense linears are shipped pre-packed as bit-exact affine-Q8 and the
        # router gates are fp16. A minimal quantization block (group/bits only,
        # no per-layer entries) is still REQUIRED: mlx-lm only builds the
        # QuantizedLinear module *structure* when `quantization` is present
        # (class_predicate matches modules that ship a `*.scales` key), then
        # `load_weights` overwrites it with the stored Q8 values verbatim. So
        # no values are re-quantized and the w8a16 contract is preserved.
        cfg["quantization"] = {"group_size": 128, "bits": 8, "mode": "affine"}
        cfg["quantization_config"] = {
            "quant_method": "eschamoe",
            "bits": 2.0,
            "format_version": "2.0",
        }
    else:
        q = {"group_size": args.group_size, "bits": args.bits, "mode": "affine"}
        n_layers = cfg["text_config"]["num_hidden_layers"]
        for l in range(n_layers):
            q[f"language_model.model.layers.{l}.mlp.switch_mlp.gate_proj"] = {"group_size": 64, "bits": 2}
            q[f"language_model.model.layers.{l}.mlp.switch_mlp.up_proj"] = {"group_size": 64, "bits": 2}
            q[f"language_model.model.layers.{l}.mlp.switch_mlp.down_proj"] = {"group_size": 64, "bits": 3}
            for m in ("mlp.gate", "mlp.shared_expert_gate"):
                q[f"language_model.model.layers.{l}.{m}"] = {"group_size": 64, "bits": args.gate_bits}
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
