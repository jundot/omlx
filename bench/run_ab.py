"""A/B campaign driver: N reps per arm with alternating (ABBA) order.

Each cell runs bench_expert_streaming.py as a FRESH subprocess — fresh cache,
fresh process state — so the only carry-over between cells is the OS page
cache, which the alternating arm order (base,treat,treat,base) is designed
to cancel on average.

Usage:
    .venv/bin/python bench/run_ab.py --out-dir bench/results/x --reps 2 \
        --shared "--model qwen-jang4m --budget auto --prompt-len short \
                  --decode 128 --single-request" \
        --arm base="" --arm pins="--pins --pin-gib 4"

An arm flag token of the form ``E:NAME=value`` sets an environment
variable on that arm's subprocess instead of a CLI flag:

        --arm nopin="E:OMLX_EXPERT_STREAMING_HOTPIN=0"

Per-cell output: <out-dir>/<model>_<prompt>_<arm>_r<rep>.json + .log
(stdout+stderr, streamed live to the console AND the log so the harness
heartbeat stays visible).
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, metavar="DIR")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--shared",
        default="",
        help="flag string shared by every arm (model, prompt-len, decode...)",
    )
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=FLAGS",
        help="one arm: name plus extra flag string (repeatable)",
    )
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        name, _, flags = spec.partition("=")
        env: dict[str, str] = {}
        keep = []
        for tok in shlex.split(flags):
            if tok.startswith("E:") and "=" in tok[2:]:
                k, v = tok[2:].split("=", 1)
                env[k] = v
            else:
                keep.append(tok)
        arms.append((name.strip(), keep, env))
    if not arms:
        raise SystemExit("at least one --arm required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shared = shlex.split(args.shared)
    model = next(
        (shared[i + 1] for i, f in enumerate(shared) if f == "--model"),
        "model",
    )
    # Stems use the basename so full model paths stay inside out-dir.
    model_stem = Path(model).name or "model"
    prompt = next(
        (shared[i + 1] for i, f in enumerate(shared) if f == "--prompt-len"),
        "short",
    )

    # ABBA: rep order alternates so warm-page-cache advantage lands on both
    # sides of the comparison equally.
    plan = []
    for rep in range(args.reps):
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for name, flags, env in order:
            plan.append((rep, name, flags, env))

    total = len(plan)
    for idx, (rep, name, flags, env) in enumerate(plan, 1):
        stem = f"{model_stem}_{prompt}_{name}_r{rep}"
        json_path = out_dir / f"{stem}.json"
        log_path = out_dir / f"{stem}.log"
        cmd = [
            sys.executable,
            str(HERE / "bench_expert_streaming.py"),
            *shared,
            *flags,
            "--out",
            str(json_path),
        ]
        print(f"=== [{idx}/{total}] {stem} ===", flush=True)
        t0 = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO),
                env={**os.environ, **env} if env else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            rc = proc.wait()
        print(
            f"--- {stem}: rc={rc} in {time.time() - t0:.0f}s ---",
            flush=True,
        )
        if rc != 0:
            print(f"ARM FAILED: {stem} (rc={rc}) — see {log_path}", flush=True)
    print(f"=== campaign done: {total} cells -> {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
