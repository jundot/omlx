# SPDX-License-Identifier: Apache-2.0
"""Offline, single-request Hy4 Flash text inference. All paths must be local."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from .artifact import Artifact, member


def render_user(artifact, text, raw, effort):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(artifact.root / "tokenizer.json"))
    if not raw:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        template = env.from_string((artifact.root / "chat_template.jinja").read_text())
        text = template.render(
            messages=[{"role": "user", "content": text}],
            tools=[],
            add_generation_prompt=True,
            reasoning_effort=effort,
        )
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise ValueError("prompt produced no tokens")
    return tokenizer, ids


def weight_shapes(model):
    from mlx.utils import tree_flatten

    return {n: tuple(w.shape) for n, w in tree_flatten(model.parameters())}


def quantize_trunk(model, trunk):
    """Swap the listed trunk linears for nn.QuantizedLinear placeholders."""
    if not trunk:
        return
    import mlx.nn as nn

    wanted, seen = set(trunk["modules"]), set()

    def pick(path, module):
        if path in wanted:
            seen.add(path)
            return True
        return False

    nn.quantize(
        model,
        group_size=trunk["group_size"],
        bits=trunk["bits"],
        mode=trunk["mode"],
        class_predicate=pick,
    )
    if seen != wanted:
        raise ValueError(f"trunk modules not in model: {sorted(wanted - seen)[:8]}")


def load_weights(model, artifact):
    """Check the complete index; evaluate one shard at a time, never all source tensors."""
    import mlx.core as mx

    from .quant_io import checkpoint_dtype, read_header

    quantize_trunk(model, artifact.config.get("trunk_quantization"))
    shapes = weight_shapes(model)
    required = artifact.weight_map.keys()
    if shapes.keys() != required:
        raise ValueError(
            f"model/checkpoint mismatch: missing {sorted(shapes.keys() - required)[:8]}; "
            f"unexpected {sorted(required - shapes.keys())[:8]}"
        )
    by_file = defaultdict(set)
    for key, shard in artifact.weight_map.items():
        by_file[shard].add(key)
    dtypes = {"U32": mx.uint32, "I32": mx.int32, "F32": mx.float32, "BF16": mx.bfloat16}
    packed_modules = set(artifact.config.get("omlx_oqh", {}).get("tensors", {}))
    packed_modules.update(
        (artifact.config.get("trunk_quantization") or {}).get("modules", [])
    )
    plan = []
    for shard, keys in sorted(by_file.items()):
        path = member(artifact.root, shard)
        with path.open("rb", buffering=0) as f:
            header = read_header(f)
        if header.keys() != keys:
            raise ValueError(f"{shard}: index and actual header disagree")
        for name, view in header.items():
            spec = view.spec
            if spec.shape != shapes[name] or spec.dtype != checkpoint_dtype(
                name, packed_modules
            ):
                raise ValueError(
                    f"{shard}: wrong checkpoint tensor type or shape: {name}"
                )
        plan.append((path, header))
    for path, header in plan:
        weights = mx.load(str(path))
        if weights.keys() != header.keys():
            raise ValueError(
                f"{path.name}: loaded tensors disagree with validated header"
            )
        for n, array in weights.items():
            if (
                tuple(array.shape) != shapes[n]
                or array.dtype != dtypes[header[n].spec.dtype]
            ):
                raise ValueError(f"wrong tensor type or shape: {n}")
        # A shard may contain only item 3 of a list whose earlier items came
        # from other shards. MLX tree_unflatten fills that sparse list with
        # empty dicts: load_weights(strict=False) can overwrite live arrays.
        # Replace validated leaves in the existing tree, never its containers.
        for name, array in weights.items():
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = parent[int(part) if isinstance(parent, list) else part]
            leaf = int(parts[-1]) if isinstance(parent, list) else parts[-1]
            parent[leaf] = array
        mx.eval(list(weights.values()))
        del weights


def prepare_fp32_head(model):
    """Materialize the reference FP32 output weight once, never cast the embedding.

    Hy4 stores its untied output head as BF16, but the reference matmul is FP32.
    Replace the in-memory parameter (not the checkpoint); steady-state growth is
    one original head. Fail closed if the model explicitly requests another mode.
    """
    import mlx.core as mx
    import mlx.nn as nn

    if model.args.tie_word_embeddings or not model.args.enable_lm_head_fp32:
        raise ValueError("precast requires an untied reference FP32 LM head")
    if isinstance(model.lm_head, nn.QuantizedLinear):
        return 0  # quantized trunk: FP32 activations, nothing to precast
    old = model.lm_head.weight
    if old.dtype == mx.float32:
        return 0
    if old.dtype != mx.bfloat16 or old.ndim != 2:
        raise ValueError("expected an untied BF16 output matrix")
    promoted = old.astype(mx.float32)
    mx.eval(promoted)
    model.lm_head.weight = promoted
    return promoted.nbytes - old.nbytes


def greedy(model, ids, *, eos, count, chunk):
    """Greedy generation with a fresh request cache and distinct prefill/decode timers."""
    import mlx.core as mx

    cache = model.make_cache()

    def advance(batch):
        last = model(mx.array([batch], dtype=mx.int32), cache=cache, logits_to_keep=1)[
            :, -1
        ]
        mx.eval(last)
        if not mx.all(mx.isfinite(last)).item():
            raise ValueError("non-finite model logits")
        return last

    start = time.monotonic()
    for i in range(0, len(ids), chunk):
        last = advance(ids[i : i + chunk])
    first = time.monotonic()
    out = []
    steps = 0
    for i in range(count):
        token = int(mx.argmax(last, axis=-1).item())
        out.append(token)
        if token in eos or i + 1 == count:
            break
        last = advance([token])
        steps += 1
    finished = time.monotonic()
    return out, {
        "prompt_tokens": len(ids),
        "new_tokens": len(out),
        "stopped_on_eos": out[-1] in eos,
        "prefill_seconds": first - start,
        "decode_steps": steps,
        "decode_seconds": finished - first,
        "decode_tokens_per_second": (
            steps / (finished - first) if steps and finished > first else None
        ),
    }


def generate(args, artifact):
    import mlx.core as mx

    from .model import Model, ModelArgs
    from .residency import open_tables

    if args.max_tokens < 1 or not 1 <= args.prefill_chunk <= 256:
        raise ValueError("invalid token count or prefill chunk")
    prepare = args.prepare_engrams
    tok, ids = render_user(
        artifact, args.prompt, args.raw_prompt, args.reasoning_effort
    )
    if len(ids) + args.max_tokens > min(
        artifact.config["max_position_embeddings"], args.context_limit
    ):
        raise ValueError("prompt and requested generation exceed the context limit")
    eos = artifact.config["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    setup = time.monotonic()
    with open_tables(
        artifact.root / "manifest.json",
        first_root=args.source,
        late_root=args.ssd_dir,
        receipt=args.receipt,
        first_mode=args.first_engram,
    ) as provider:
        model = Model(ModelArgs.from_dict(artifact.config))
        model.eval()
        load_weights(model, artifact)
        precast = args.precast_lm_head
        head_growth = prepare_fp32_head(model) if precast else 0
        model.set_engram_tables(provider)
        if prepare:
            from .prepared_engram import install_prepared_engram

            preparation = install_prepared_engram(model)
        else:
            preparation = nullcontext()
        with preparation:
            setup_seconds = time.monotonic() - setup
            tokens, metrics = greedy(
                model, ids, eos=eos, count=args.max_tokens, chunk=args.prefill_chunk
            )
        text = tok.decode(tokens, skip_special_tokens=True)
        result = {
            "text": text,
            "token_ids": tokens,
            "load_and_wire_seconds": setup_seconds,
            **metrics,
            "engram": provider.stats(),
            "mlx_peak_bytes": mx.get_peak_memory(),
            "prepare_engrams": prepare,
            "precast_lm_head": precast,
            "lm_head_extra_resident_bytes": head_growth,
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        else:
            print(text)
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("text", "token_ids")}
                ),
                file=sys.stderr,
            )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["stage-tables"]:
        from .stage import main as stage

        return stage(argv[1:])
    if argv[:1] == ["convert"]:
        from .convert import main as convert

        return convert(argv[1:])
    p = argparse.ArgumentParser(prog="omlx hy4-flash", description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser(
        "convert",
        help="stream and verify an oQh text export (8-bit default, --bits 4/5/6, --trunk-bits 8)",
    )
    subs.add_parser("stage-tables", help="verify and stage only the three late tables")
    for cmd, helptext in [
        ("inspect", "check the local oQh export"),
        ("verify-tables", "authenticate the four BF16 source files"),
        ("generate", "run a single text generation"),
    ]:
        a = subs.add_parser(cmd, help=helptext)
        a.add_argument(
            "--model", required=True, type=Path, help="converted native oQh directory"
        )
        a.add_argument(
            "--verify-core",
            action="store_true",
            help="also hash every core shard in full",
        )
        if cmd in ("verify-tables", "generate"):
            a.add_argument(
                "--source",
                required=True,
                type=Path,
                help="original verified BF16 snapshot (first table)",
            )
            a.add_argument(
                "--ssd-dir",
                required=True,
                type=Path,
                help="internal SSD directory containing the three verified late shards",
            )
            a.add_argument(
                "--receipt",
                required=True,
                type=Path,
                help="private local table verification receipt",
            )
        if cmd == "generate":
            a.add_argument("--prompt", required=True)
            a.add_argument("--raw-prompt", action="store_true")
            a.add_argument(
                "--reasoning-effort",
                choices=["no_think", "high", "max"],
                default="no_think",
            )
            a.add_argument("--max-tokens", type=int, default=32)
            a.add_argument("--prefill-chunk", type=int, default=16)
            a.add_argument(
                "--context-limit",
                type=int,
                default=4096,
                help="request guard, not architecture context size",
            )
            a.add_argument(
                "--prepare-engrams",
                action="store_true",
                help="prepare token-only Engram gathers before the decoder layers (opt-in)",
            )
            a.add_argument(
                "--first-engram",
                choices=("mlock", "pread"),
                default="pread",
                help="read the first Engram table on demand (default) or explicitly wire it",
            )
            a.add_argument(
                "--precast-lm-head",
                action="store_true",
                help="keep the reference FP32 vocabulary projection resident (+0.69 GiB)",
            )
            a.add_argument(
                "--json",
                action="store_true",
                help="include token IDs, performance and live residency counters",
            )
    args = p.parse_args(argv)
    try:
        a = Artifact.open(args.model, verify_core=args.verify_core)
        if args.command == "inspect":
            print(json.dumps(a.summary(), indent=2))
        elif args.command == "verify-tables":
            from .residency import prepare_table_receipt

            prepare_table_receipt(
                a.root / "manifest.json",
                first_root=args.source,
                late_root=args.ssd_dir,
                receipt=args.receipt,
            )
        elif args.command == "generate":
            generate(args, a)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        p.exit(1, f"hy4-flash: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
