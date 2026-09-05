"""Perplexity harness for expert-streaming checkpoints (Fase I4).

Computes token NLL / perplexity over a local corpus for any MLX (quantized)
checkpoint via mlx_lm's resident path. Streaming compute is bit-exact versus
the resident path (test-pinned in tests/test_expert_streaming.py), so the
resident measurement is representative — fast, and without touching the SSD
streaming machinery.

Primary purpose: the quality gate for precision changes (the Fase I5 cold
tier) — compare oQ4e vs oQ2.7-cold-tier checkpoints on the same corpus.

Usage:
    .venv/bin/python bench/ppl_expert_streaming.py \
        --model "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e" \
        --corpus corpus.txt --max-windows 64 --out bench/results/ppl_glm.json

    # streaming mode — for checkpoints whose expert banks far exceed RAM
    # (loads through the omlx expert-streaming engine, same as the server):
    .venv/bin/python bench/ppl_expert_streaming.py --streaming \
        --model glm --cold-tier none --budget 2.0 --ctx 1024 --max-windows 24 \
        --corpus bench/corpus/pg1342.txt --out bench/results/ppl_runs/glm_base.json
    .venv/bin/python bench/ppl_expert_streaming.py --streaming \
        --model glm --cold-tier 3 --budget 2.0 --ctx 1024 --max-windows 24 \
        --corpus bench/corpus/pg1342.txt --out bench/results/ppl_runs/glm_cold3.json

Corpus: a plain UTF-8 text file (one or more documents; whitespace-joined).
Windows are disjoint (no overlap) of --ctx tokens each; mean NLL over
predicted tokens only (the first token of each window is context).

The streaming cold-tier arm is the real shipped path: experts are read from
`<model>/expert_cold/` and dequantized with the tier bits recorded in the
shard metadata.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATHS = {
    "qwen": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-oQ4e-mtp",
    "qwen-jang": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4S",
    "qwen-jang4m": "/Volumes/SSD 4TB/AI Models/Qwen3.8-Flash-Next-JANG_4M",
    "glm": "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e",
    "dsv4": "/Volumes/SSD 4TB/AI Models/DeepSeek-V4-Flash-0731-oQ4e-mtp",
}


def iter_windows(token_ids: list[int], ctx: int, max_windows: int | None):
    """Disjoint [ctx]-token windows. Yields (start, window). The first token
    of each window is context only — it produces no NLL term."""
    step = ctx - 1
    yielded = 0
    start = 0
    while start + ctx <= len(token_ids):
        yield start, token_ids[start : start + ctx]
        yielded += 1
        if max_windows is not None and yielded >= max_windows:
            break
        start += step


def window_nll(logits: "np.ndarray", targets: "np.ndarray") -> tuple[float, int]:
    """Mean NLL of `targets` under `logits` ([ctx, vocab] float32), skipping
    the first position (context token). Returns (sum_nll, n_terms)."""
    # logits[:-1] predict targets[1:]
    lg = logits[:-1].astype(np.float32)
    tg = targets[1:].astype(np.int64)
    lg = lg - lg.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(lg).sum(axis=-1))
    picked = lg[np.arange(len(tg)), tg]
    nll = logsumexp - picked
    return float(nll.sum()), int(len(tg))


def run_streaming(model_path: str, text: str, args) -> dict:
    """Load through the omlx expert-streaming engine (models whose expert
    banks far exceed RAM) and run the same disjoint-window NLL over raw
    forwards. Streaming compute is bit-exact with the resident path
    (test-pinned in tests/test_expert_streaming.py). With --cold-tier the
    engine's backing store routes expert reads to `<model>/expert_cold/`
    exactly as the server would."""
    import asyncio

    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_expert_streaming import FakeEnforcer

    async def _run() -> dict:
        from mlx_lm.models.cache import make_prompt_cache

        from omlx.engine_pool import EnginePool
        from omlx.model_settings import ModelSettings
        from omlx.scheduler import SchedulerConfig
        from omlx.utils.proc_memory import get_phys_footprint

        model_dir = Path(model_path)
        entry_name = model_dir.name
        pool = EnginePool(scheduler_config=SchedulerConfig(hot_cache_max_size=0))
        pool._process_memory_enforcer = FakeEnforcer(args.mem_ceiling_gib)
        pool.discover_models(str(model_dir.parent))
        if pool.get_entry(entry_name) is None:
            raise SystemExit(f"engine pool has no entry for {entry_name}")

        settings = ModelSettings(
            expert_streaming_enabled=True,
            expert_streaming_budget_gib=args.budget,
            expert_streaming_cache_prior=args.cache_prior,
            expert_streaming_topk_threshold=(
                None if args.topk is None or args.topk >= 1.0 else args.topk
            ),
            expert_streaming_cold_tier=(
                None if args.cold_tier == "none" else args.cold_tier
            ),
            expert_streaming_hot_fraction=(
                None if not args.hot_fraction else float(args.hot_fraction)
            ),
            expert_streaming_pins=True,
            expert_streaming_pin_gib=1.25,
            qwen4_ple_ssd_offload=True,
        )
        t0 = time.perf_counter()
        engine = await pool.get_engine(entry_name, runtime_settings=settings)
        t_load = time.perf_counter() - t0
        print(
            f"engine loaded in {t_load:.1f}s "
            f"phys {get_phys_footprint() / 1024**3:.2f}G",
            flush=True,
        )

        # Same watermarks the server's ProcessMemoryEnforcer propagates;
        # without them the scheduler's prefill guard never engages.
        try:
            sched = getattr(
                getattr(getattr(pool.get_entry(entry_name).engine, "_engine", None), "engine", None),
                "scheduler",
                None,
            )
            if sched is not None:
                gib = 1024**3
                sched._memory_hard_limit_bytes = int(args.mem_ceiling_gib * gib)
                sched._memory_limit_bytes = int(args.mem_ceiling_gib * 0.9 * gib)
                sched._memory_abort_limit_bytes = int(args.mem_ceiling_gib * 0.95 * gib)
                print(
                    f"scheduler memory limits: hard {args.mem_ceiling_gib:.0f}G "
                    f"soft {args.mem_ceiling_gib * 0.9:.0f}G",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"scheduler limit setup skipped: {e}")

        tokenizer = engine.tokenizer
        vlm_model = getattr(engine, "_vlm_model", None)
        lm = getattr(vlm_model, "language_model", None) or vlm_model

        if os.environ.get("OMLX_PPL_DISABLE_STREAM_EVAL") == "1":
            # Diagnostic only: the per-layer eval/clear_cache boundary is
            # bit-exact (test-pinned), so NLL must not change. If it does,
            # the boundary itself is implicated at this sequence length.
            layers = None
            for path in [
                ("language_model", "model", "layers"),
                ("language_model", "layers"),
                ("model", "layers"),
                ("layers",),
            ]:
                cur = vlm_model
                ok = True
                for a in path:
                    if not hasattr(cur, a):
                        ok = False
                        break
                    cur = getattr(cur, a)
                if ok and cur is not None and len(cur) > 0:
                    layers = cur
                    break
            n_off = 0
            for layer in layers or []:
                if getattr(layer, "_stream_eval", False):
                    layer._stream_eval = False
                    n_off += 1
            print(f"diagnostic: disabled stream_eval on {n_off} layers", flush=True)

        ids = tokenizer.encode(text)
        windows = list(iter_windows(ids, args.ctx, args.max_windows))
        if not windows:
            raise SystemExit(
                f"corpus too short: {len(ids)} tokens < one {args.ctx}-token window"
            )
        print(
            f"{len(ids)} tokens -> {len(windows)} disjoint {args.ctx}-token windows",
            flush=True,
        )

        # Fase 0: instrument the MoE router. mlx nn.Module resolves
        # __call__ on the class only (instance attribute assignment does not
        # shadow it), so we compose a class-level wrapper over the current
        # (engine-patched) __call__ — exactly how the oMLX patches chain.
        # Per call we recompute the gate GEMM (hidden x num_experts, tiny)
        # to capture pre-softmax logits plus argpartition top-k/scores.
        capture_states = []
        if args.capture_routing:
            import mlx_vlm.models.qwen3_5_moe.language as moe_mod

            moe_cls = moe_mod.Qwen3_5MoeSparseMoeBlock
            if getattr(moe_cls, "_omlx_routing_capture", False):
                raise SystemExit("routing capture already installed")
            capture_state_by_id: dict[int, dict] = {}
            _capture_orig_call = moe_cls.__call__

            def _capture_patched(self, x, *a, **kw):
                out = _capture_orig_call(self, x, *a, **kw)
                st = capture_state_by_id.get(id(self))
                if st is not None:
                    g = self.gate(x)
                    gs = mx.softmax(g, axis=-1, precise=True)
                    k = st["k"]
                    inds = mx.argpartition(gs, kth=-k, axis=-1)[..., -k:]
                    sc = mx.take_along_axis(gs, inds, axis=-1)
                    st["logits"].append(g)
                    st["inds"].append(inds)
                    st["scores"].append(sc)
                return out

            moe_cls.__call__ = _capture_patched
            moe_cls._omlx_routing_capture = True

            layers = None
            for path in [
                ("language_model", "model", "layers"),
                ("language_model", "layers"),
                ("model", "layers"),
                ("layers",),
            ]:
                cur = vlm_model
                ok = True
                for a in path:
                    if not hasattr(cur, a):
                        ok = False
                        break
                    cur = getattr(cur, a)
                if ok and cur is not None and len(cur) > 0:
                    layers = cur
                    break
            for li, layer in enumerate(layers or []):
                moe = getattr(layer, "mlp", None)
                if moe is None or not hasattr(moe, "gate") or not hasattr(moe, "top_k"):
                    continue
                capture_state_by_id[id(moe)] = {
                    "layers": li,
                    "k": int(moe.top_k),
                    "logits": [],
                    "inds": [],
                    "scores": [],
                }
            capture_states = list(capture_state_by_id.values())
            print(
                f"routing capture: {len(capture_states)} MoE layers instrumented "
                f"(class {moe_cls.__name__})",
                flush=True,
            )

        total_nll = 0.0
        total_terms = 0
        for i, (_, window) in enumerate(windows):
            t_w = time.perf_counter()
            arr = mx.array(np.array([window], dtype=np.int32))
            # Fresh cache per window (windows are disjoint). The glm5_next
            # decoder produces garbage without a cache — the scheduler always
            # passes one (self.model(chunk, cache=state.cache)).
            cache = make_prompt_cache(lm)
            logits = lm(arr, cache=cache)
            if hasattr(logits, "logits"):  # mlx_vlm LanguageModelOutput
                logits = logits.logits
            elif isinstance(logits, tuple):
                logits = logits[0]
            nll, n = window_nll(
                np.array(logits[0].astype(mx.float32)), np.array(window)
            )
            total_nll += nll
            total_terms += n
            mx.clear_cache()
            ppl_so_far = math.exp(total_nll / total_terms)
            print(
                f"  window {i + 1}/{len(windows)}  running ppl {ppl_so_far:.4f}"
                f"  ({time.perf_counter() - t_w:.1f}s)",
                flush=True,
            )

        # Persist the learned pin profile (the server does this in stop();
        # the harness tears down via release/unload, so save explicitly —
        # Fase I6's HOBBIT split needs the frequencies on the next load).
        from omlx.patches.expert_streaming import save_expert_pin_profile

        save_expert_pin_profile(engine)

        if capture_states:
            outdir = Path(args.capture_routing)
            outdir.mkdir(parents=True, exist_ok=True)
            saved = 0
            first_tokens = None
            first_k = None
            for state in capture_states:
                if not state["logits"]:
                    continue
                lg = mx.concatenate(state["logits"], axis=0).astype(mx.float16)
                ids = mx.concatenate(state["inds"], axis=0).astype(mx.int16)
                sc = mx.concatenate(state["scores"], axis=0).astype(mx.float16)
                mx.eval(lg, ids, sc)
                if first_tokens is None:
                    first_tokens = lg.shape[0]
                    first_k = state["k"]
                np.savez(
                    outdir / f"layer_{state['layers']:04d}.npz",
                    logits=np.array(lg),
                    topk=np.array(ids),
                    scores=np.array(sc),
                    k=np.array(state["k"], dtype=np.int8),
                )
                saved += 1
            (outdir / "meta.json").write_text(
                json.dumps(
                    {
                        "model": model_path,
                        "corpus": str(args.corpus),
                        "ctx": args.ctx,
                        "windows": len(windows),
                        "tokens": first_tokens,
                        "n_layers": saved,
                        "k": first_k,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"wrote routing captures ({saved} layers) to {outdir}", flush=True)

        await pool.release_engine(entry_name)
        await pool._unload_engine(entry_name)

        mean_nll = total_nll / total_terms
        # Fase 0 deliverable: per-token disk-read census (precondition for
        # ADR-0001 D6 Fase 3). ru_inblock counts 512-byte blocks actually
        # read from disk by THIS process (page-cache hits do not count).
        import resource as _res

        _ru = _res.getrusage(_res.RUSAGE_SELF)
        io_gb = _ru.ru_inblock * 512 / 1e9
        io_per_tok = io_gb * 1e9 / max(total_terms, 1)
        io_per_win = io_gb / max(len(windows), 1)
        return {
            "model": model_path,
            "corpus": str(args.corpus),
            "ctx": args.ctx,
            "windows": len(windows),
            "n_terms": total_terms,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "mode": "streaming",
            "cold_tier": args.cold_tier,
            "hot_fraction": args.hot_fraction,
            "budget_gib": args.budget,
            "load_s": round(t_load, 1),
            "disk_read_gb": round(io_gb, 3),
            "disk_read_mb_per_window": round(io_per_win * 1024, 1),
            "disk_read_bytes_per_term": round(io_per_tok, 1),
        }

    return asyncio.run(_run())


def main() -> None:
    # INFO logs (streaming conversion, HOBBIT split, cold tier) are gate
    # evidence — the "HOBBIT split on 42/42 layers (fraction 0.25)" line the
    # I6 gate requires must be in the harness log.
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help="checkpoint path or bench alias (qwen/glm/dsv4)",
    )
    ap.add_argument("--corpus", required=True, help="plain UTF-8 text file")
    ap.add_argument("--ctx", type=int, default=2048, help="window size in tokens")
    ap.add_argument(
        "--max-windows", type=int, default=64, help="stop after this many windows"
    )
    ap.add_argument("--out", default=None, help="write a JSON result here")
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="load through the omlx expert-streaming engine (for checkpoints "
        "whose expert banks far exceed RAM; --cold-tier selects the arm)",
    )
    ap.add_argument(
        "--hot-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help="HOBBIT split fraction (expert_streaming_hot_fraction; needs a "
        "learned pin profile from --cold-tier runs)",
    )
    ap.add_argument(
        "--cold-tier",
        choices=["none", "2", "3"],
        default="none",
        help="expert_streaming_cold_tier for the streaming load (streaming mode only)",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=2.0,
        metavar="GIB",
        help="streaming cache budget (streaming mode only)",
    )
    ap.add_argument(
        "--cache-prior",
        type=float,
        default=None,
        metavar="BONUS",
        help="cache-prior logit bonus for resident experts (streaming mode "
        "only; approximate routing quality gate)",
    )
    ap.add_argument(
        "--topk",
        type=float,
        default=None,
        metavar="MASS",
        help="adaptive top-k mass threshold (streaming mode only; approximate "
        "routing quality gate — matches bench_expert_streaming.py --topk)",
    )
    ap.add_argument(
        "--mem-ceiling-gib",
        type=float,
        default=14.0,
        metavar="GIB",
        help="scheduler memory ceiling for the streaming load",
    )
    ap.add_argument(
        "--capture-routing",
        default=None,
        metavar="DIR",
        help="write per-layer MoE router captures (logits/topk/scores) to DIR "
        "(Fase 0: routing fidelity baselines)",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=12.0,
        metavar="GB",
        help="abort when available memory is below this (ppl is latency-"
        "insensitive so this can be lower than the tok/s bench floor)",
    )
    args = ap.parse_args()

    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"aborted: only {free_gb:.1f} GB available (need {args.min_free_gb:.0f}+)"
            )
        print(f"memory preflight ok: {free_gb:.1f} GB available", flush=True)
    except ImportError:
        pass

    model_path = MODEL_PATHS.get(args.model, args.model)
    text = Path(args.corpus).read_text(encoding="utf-8")
    if not text.strip():
        print("corpus is empty", file=sys.stderr)
        sys.exit(2)

    if args.streaming:
        result = run_streaming(model_path, text, args)
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
            print(f"wrote {args.out}")
        return

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    import mlx.core as mx

    print(f"loading {model_path} ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ids = tokenizer.encode(text)
    windows = list(iter_windows(ids, args.ctx, args.max_windows))
    if not windows:
        print(
            f"corpus too short: {len(ids)} tokens < one {args.ctx}-token window",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"{len(ids)} tokens -> {len(windows)} disjoint {args.ctx}-token windows",
        flush=True,
    )

    total_nll = 0.0
    total_terms = 0
    for i, (_, window) in enumerate(windows):
        arr = mx.array(np.array([window], dtype=np.int32))
        cache = make_prompt_cache(model)
        logits = model(arr, cache=cache)
        # np has no bfloat16 — promote before the numpy NLL.
        nll, n = window_nll(
            np.array(logits[0].astype(mx.float32)), np.array(window)
        )
        total_nll += nll
        total_terms += n
        ppl_so_far = math.exp(total_nll / total_terms)
        print(
            f"  window {i + 1}/{len(windows)}  running ppl {ppl_so_far:.4f}",
            flush=True,
        )

    mean_nll = total_nll / total_terms
    result = {
        "model": model_path,
        "corpus": str(args.corpus),
        "ctx": args.ctx,
        "windows": len(windows),
        "n_terms": total_terms,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
